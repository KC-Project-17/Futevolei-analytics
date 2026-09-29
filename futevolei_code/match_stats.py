"""
Estadísticas de futevôlei a partir de:
  - rallies            : salida de rallies.detect_rallies (o las corregidas en la etapa 2)
  - traj               : salida de ball_trajectory.build_ball_trajectory
  - df_det             : detecciones limpias (pelota + jugadores)
  - homographies       : {frame: H} de rally_homographies.compute_homographies_in_rallies
                         (imagen -> cancha 1800x900, red en x=900)
  - meta               : fps, width, height, n_frames
  - cam_motion         : salida de camera.camera_motion

SIN mirar el marcador. El ganador de cada punto se deduce de:
  - dónde cae la pelota (dentro / fuera de la cancha y de qué lado), y
  - quién la tocó por última vez.
Si no se vio la caída, se usa la regla del juego: quien gana el punto saca el
siguiente. Los equipos cambian de lado cada 6 puntos (suma de ambos), como dice el
reglamento; se puede corregir a mano con side_switches=[...].

    stats = compute_statistics(rallies, traj, detections_clean, homographies, meta, cam_motion=cam_motion)
    stats["rally_stats"] -> una fila por jugada
    stats["shots"]       -> una fila por saque / ataque, con su resultado (punto / defendido / fuera)
    stats["summary"]     -> una fila por equipo
"""
import numpy as np
import pandas as pd

from .ball_sides import clean_sides, crossings_from_sides, detect_touches, side_per_frame
from .camera import compensate_pan, net_from_pan, net_from_players
from .common import meta_to_dict
from .court_geometry import COURT_H, COURT_W, NET_X, SIDE_LEFT, SIDE_RIGHT, nearest_homography, other_side, to_court
from .rally_homographies import stabilize_homographies


def compute_statistics(rallies, traj, df_det, homographies, meta, cam_motion=None,
                          line_margin=40, dv_min=650, max_dist_h=15, verbose=True,
                          serve_guard_sec=0.6, min_side_no_touch_sec=1.0,
                          serve_side_max_sec=0.7, closeup_h=0.40, use_players=True,
                          winner_by_next_serve=True, switch_every=6, prior_points=0, side_switches=None,
                          end_extra_sec=0.3, next_serve_first=False,
                          stabilize_h=True,
                          control_min_touches=2, control_min_sec=2.0):
    """
    Estadísticas por jugada, por golpe y por equipo, sin mirar el marcador.

    Args:
        rallies (pd.DataFrame): jugadas (detect_rallies o las corregidas en la etapa 2).
        traj (pd.DataFrame): trayectoria de la pelota (build_ball_trajectory).
        df_det (pd.DataFrame): detecciones limpias (pelota + jugadores).
        homographies (dict | None): {frame: H} imagen -> cancha 1800x900.
        meta (dict | pd.DataFrame): metadatos del video.
        cam_motion (pd.DataFrame | None): salida de camera_motion(video_path, rallies, meta).
            Muy recomendado: con él se sabe dónde está la red aunque la cámara gire o la red
            quede fuera de imagen. Sin él se usa la homografía y, si no hay, la red estimada
            con los jugadores.
        line_margin (float): unidades de cancha (1800x900) de tolerancia en las líneas.
        dv_min (float): cambio de velocidad mínimo para un toque (ver detect_touches).
        max_dist_h (int): distancia máxima (frames) para usar la H más cercana.
        verbose (bool): imprime el resumen.
        serve_guard_sec (float): segundos después del saque en los que no se buscan toques.
        min_side_no_touch_sec (float): tramos de lado sin toques más cortos que esto son ruido.
        serve_side_max_sec (float): el lado del saque solo se toma si se ve antes de este tiempo.
        closeup_h (float): altura relativa de jugador a partir de la cual es primer plano
            (esas homografías se descartan).
        use_players (bool): usar la red estimada por jugadores como último recurso.
        winner_by_next_serve (bool): usar la regla "quien gana saca la siguiente".
        switch_every (int | None): los equipos cambian de lado cada vez que la suma de puntos
            es múltiplo de este número (reglamento: 6). None = no cambian.
        prior_points (int): puntos jugados antes de la primera jugada del video (0 si empieza 0-0).
        side_switches (list | None): rally_id de las jugadas que se juegan YA con los lados
            cambiados (p. ej. [7, 13]). Si se pasa, reemplaza a la regla de switch_every (útil
            si la transmisión se salteó algún punto y la cuenta quedó corrida).
        end_extra_sec (float): en jugadas que terminan en 'ground', segundos después del contacto
            en los que se sigue mirando de qué lado está la pelota (ataque que pica pegado a la red).
        next_serve_first (bool): False (por defecto) = el ganador sale de dónde cae la pelota
            (homografía) cuando se ve la caída; si no, de quién saca la jugada siguiente.
            True = quién saca la siguiente siempre que se sepa; la caída solo como respaldo.
        stabilize_h (bool): con cam_motion, combina las homografías de cada plano
            (stabilize_homographies): hay H en todos los frames del plano y se descartan las
            sueltas malas.
        control_min_touches (int): si después del último cruce visto el equipo que recibió tocó
            la pelota >= control_min_touches veces (o la tuvo >= control_min_sec), se considera
            que la controló y que la jugada terminó con SU ataque (que no se vio cruzar o que se
            fue a la red / afuera).
        control_min_sec (float): ver control_min_touches.

    Returns:
        dict: DataFrames
            'rally_stats' : una fila por jugada (quién sacó, cómo terminó el saque, cuántos ataques
                        hizo cada equipo, cómo terminó la jugada y quién ganó)
            'shots'       : una fila por saque / ataque con su resultado (punto / defendido / fuera)
            'summary'     : una fila por equipo
    """
    meta = meta_to_dict(meta)
    fps = meta["fps"]
    homographies = homographies or {}
    cx = traj["cx"].to_numpy()
    cy = traj["cy"].to_numpy()
    n = len(traj)

    # Primeros planos (jugador grande en pantalla): no se ve la cancha y el modelo de
    # keypoints inventa homografías -> se descartan (en este partido eran las del saque
    # filmado de cerca y movían la red ~800 px)
    _p = df_det[df_det["type"] == "player"]
    _h = ((_p["y2"] - _p["y1"]) / meta["height"]).groupby(_p["frame"]).max().reindex(range(n))
    closeup = _h.rolling(max(1, int(0.3 * fps)), center=True, min_periods=1).median().fillna(0).to_numpy() > closeup_h
    n_h0 = len(homographies)
    homographies = {f: H for f, H in homographies.items() if 0 <= f < n and not closeup[f]}
    if verbose and n_h0 - len(homographies):
        print(f"Homografías descartadas por estar en primer plano: {n_h0 - len(homographies)}")
    if stabilize_h and cam_motion is not None and len(cam_motion) and homographies:
        homographies = stabilize_homographies(homographies, cam_motion, fps=fps, verbose=verbose)
    frames_h = np.array(sorted(homographies.keys()))

    net_pan, traj_t, pan_margin = None, traj, 80.0
    if cam_motion is not None and len(cam_motion):
        net_pan, shots_info = net_from_pan(cam_motion, df_det, meta, homographies, closeup_h=closeup_h, verbose=verbose)
        # margen según la precisión de la referencia: H ~±50 px, jugadores ~±100-250 px
        pan_margin = np.full(n, 80.0)
        for _, pl in shots_info.iterrows():
            if pl.source in ("jugadores", "hueco_jugadores"):
                pan_margin[int(pl.start):int(pl.end) + 1] = 150.0
        traj_t = compensate_pan(traj, cam_motion, meta)

    # Cambios de cámara: cerca de ellos la velocidad de la pelota "salta" (no es un toque)
    view_edges = list(np.where(np.diff(closeup.astype(int)) != 0)[0] + 1)
    if cam_motion is not None and len(cam_motion):
        view_edges += cam_motion.loc[cam_motion["cut"], "frame"].astype(int).tolist()
    near_cut = np.zeros(n, dtype=bool)
    for f_ in view_edges:
        near_cut[max(0, f_ - int(0.15 * fps)):min(n, f_ + int(0.3 * fps))] = True

    net_players = net_from_players(df_det, meta) if use_players else None

    rows_out, info = [], {}
    for _, r in rallies.iterrows():
        rid, s0, e0 = int(r["rally_id"]), int(r["serve_frame"]), int(r["end_frame"])
        # Si la jugada terminó con la pelota en la arena, se mira un poco después del contacto:
        # un ataque que pica justo pasando la red se ve del otro lado recién ahí
        e_ext = min(n - 1, e0 + int(end_extra_sec * fps)) if r["end_reason"] == "ground" else e0
        sides = side_per_frame(traj, s0, e_ext, homographies, frames_h, meta, max_dist_h=max_dist_h,
                               net_players_x=net_players, net_pan_x=net_pan, pan_margin_px=pan_margin)
        ref_source = side_per_frame.ref_source[:e0 - s0 + 1]
        if e_ext > e0:
            end_t = sides[-1]
            sides = sides[:e0 - s0 + 1]
            if end_t is not None and sides[-1] is not None and end_t != sides[-1]:
                sides[-1] = end_t
        nf = max(1, len(ref_source))

        # Lado del saque: el primero conocido, solo si aparece poco después del saque
        k0 = next((k for k, l in enumerate(sides) if l is not None), None)
        serve_side = sides[k0] if (k0 is not None and k0 <= serve_side_max_sec * fps) else None

        # Toques (sin el vuelo del saque ni los pegados a un cambio de cámara)
        tq = [s0] + [t for t in detect_touches(traj_t, s0, e0, meta, dv_min=dv_min)
                     if t > s0 + int(serve_guard_sec * fps) and not near_cut[t]]

        # Tramos de lado cortos y sin toques = ruido cerca de la red
        sides = clean_sides(sides, s0, tq, int(min_side_no_touch_sec * fps))
        crossings = crossings_from_sides(sides, s0)
        tq_side = [sides[t - s0] if s0 <= t <= e0 else None for t in tq]

        # Lado del saque no visto (saque en primer plano): el primer toque es la recepción
        # del rival -> el saque vino del otro lado
        if serve_side is None:
            receiver_side = next((l for l in tq_side[1:] if l is not None), None)
            if receiver_side is not None:
                serve_side = other_side(receiver_side)
            else:
                last_pos = next((l for l in reversed(sides) if l is not None), None)
                serve_side = other_side(last_pos) if last_pos is not None else None
            k0 = next((k for k, l in enumerate(sides) if l is not None), None)
            if serve_side is not None and k0 is not None and k0 > 0:
                sides[:k0] = [serve_side] * k0
                crossings = crossings_from_sides(sides, s0)
                tq_side = [sides[t - s0] if s0 <= t <= e0 else None for t in tq]

        # Posesiones: tramos entre cruces. La primera es la del saque.
        bounds = [s0] + [c[0] for c in crossings] + [e0 + 1]
        possessions = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            side = sides[min(a - s0, len(sides) - 1)] if i > 0 else serve_side
            possessions.append({"side": side, "start": a, "end": b - 1,
                               "touches": [t for t in tq if a <= t < b]})

        # ---- Ganador del punto (sin marcador) ----
        last_touch_side = next((tq_side[k] for k in range(len(tq) - 1, -1, -1) if tq_side[k]), None)
        final_side = next((l for l in reversed(sides) if l is not None), None)
        landing = None
        winner_side, method = None, "desconocido"
        H_end = nearest_homography(homographies, frames_h, e0, max_dist_h) if len(frames_h) else None
        if r["end_reason"] == "ground" and H_end is not None and not np.isnan(cx[e0]):
            xc, yc = to_court(H_end, cx[e0], cy[e0])
            landing = (xc, yc)
            inside = (-line_margin <= xc <= COURT_W + line_margin) and (-line_margin <= yc <= COURT_H + line_margin)
            landing_side = SIDE_LEFT if xc < NET_X else SIDE_RIGHT
            if inside:
                winner_side, method = other_side(landing_side), "cae_dentro"
            elif last_touch_side is not None:
                winner_side, method = other_side(last_touch_side), "cae_fuera"
        if winner_side is None and last_touch_side is not None:
            if final_side is not None and final_side != last_touch_side:
                winner_side, method = last_touch_side, "estimado_cruzo_y_no_volvio"
            else:
                winner_side, method = other_side(last_touch_side), "estimado_no_cruzo"

        rows_out.append({
            "rally_id": rid, "serve_frame": s0, "end_frame": e0, "end_reason": r["end_reason"],
            "serve_side": serve_side, "n_touches": len(tq), "n_crossings": len(crossings),
            "coverage_pan": round(sum(src_k == "paneo" for src_k in ref_source) / nf, 2),
            "coverage_h": round(sum(src_k == "H" for src_k in ref_source) / nf, 2),
            "coverage_players": round(sum(src_k == "jugadores" for src_k in ref_source) / nf, 2),
            "landing_x": None if landing is None else round(landing[0], 1),
            "landing_y": None if landing is None else round(landing[1], 1),
            "winner_side": winner_side, "winner_method": method,
        })
        info[rid] = {"possessions": possessions, "crossings": crossings, "s0": s0, "e0": e0}

    rdf = pd.DataFrame(rows_out)
    if rdf.empty:
        if verbose:
            print("No hay jugadas para analizar.")
        return {"rally_stats": rdf, "shots": pd.DataFrame(), "summary": pd.DataFrame()}
    # Equipos por lado, con cambio de lado cada `switch_every` puntos (suma de ambos equipos)
    rdf = _assign_teams(rdf, switch_every, prior_points, side_switches)
    # Ganador por la regla del juego: quien gana el punto saca el siguiente.
    # Se usa cuando no se vio la caída. Se razona con EQUIPOS (no lados) para que
    # funcione también justo antes de un cambio de lado.
    if winner_by_next_serve:
        next_team = rdf["serving_team"].shift(-1)
        for k in rdf.index:
            if not isinstance(next_team[k], str):
                continue
            if not next_serve_first and rdf.at[k, "winner_method"] in ("cae_dentro", "cae_fuera"):
                continue
            rdf.at[k, "winner_side"] = SIDE_LEFT if rdf.at[k, "team_left"] == next_team[k] else SIDE_RIGHT
            rdf.at[k, "winner_method"] = "saque_siguiente"
    rdf["winner_team"] = [(a if g == SIDE_LEFT else b) if isinstance(g, str) else None
                             for g, a, b in zip(rdf.winner_side, rdf.team_left, rdf.team_right)]
    shots = _shots_with_result(info, rdf, fps, control_min_touches, control_min_sec)
    rdf = _rallies_with_shots(rdf, shots)
    summary = _team_summary(rdf, shots)
    if verbose:
        print(f"Jugadas: {len(rdf)} | saques: {int((shots.shot_type == 'saque').sum())} | "
              f"ataques: {int((shots.shot_type == 'ataque').sum())} "
              f"({int((~shots.crossing_seen).sum())} deducidos sin ver el cruce)")
        print(f"Ganador: {(rdf.winner_method == 'cae_dentro').sum()} por caída dentro, "
              f"{(rdf.winner_method == 'cae_fuera').sum()} por caída fuera, "
              f"{(rdf.winner_method == 'saque_siguiente').sum()} por quién saca la siguiente, "
              f"{rdf.winner_method.str.startswith('estimado').sum()} estimados")
        no_info = rdf[(rdf.coverage_pan + rdf.coverage_h + rdf.coverage_players) < 0.3]
        if len(no_info):
            print(f"⚠️  {len(no_info)} jugadas casi sin información de lado (poco fiables): {no_info.rally_id.tolist()}")
    return {"rally_stats": rdf, "shots": shots, "summary": summary}


def _assign_teams(rdf, switch_every=6, prior_points=0, side_switches=None):
    """
    Equipo 1 = el que saca la primera jugada. Los equipos cambian de lado cada vez que
    la suma de puntos llega a un múltiplo de `switch_every` (reglamento: 6; en el 3er set
    de una final, 5). Se asume 1 punto por jugada detectada y `prior_points` puntos
    jugados antes de la primera jugada del video (0 si el video empieza 0-0).
    Ojo: si la transmisión no muestra algún punto, desde ahí los cambios de lado quedan
    corridos una jugada (se puede corregir con side_switches o revisando 'side_switch').

    Args:
        rdf (pd.DataFrame): una fila por jugada con rally_id y serve_side.
        switch_every (int | None): cada cuántos puntos cambian de lado (None = no cambian).
        prior_points (int): puntos jugados antes de la primera jugada del video.
        side_switches (list | None): rally_id donde empiezan los lados cambiados (reemplaza la regla).

    Returns:
        pd.DataFrame: copia de rdf con team_left, team_right, side_switch y serving_team.
    """
    rdf = rdf.copy()
    n = len(rdf)
    total = prior_points + np.arange(n)                       # puntos jugados antes de cada jugada
    if side_switches is not None:
        n_flips = np.cumsum(rdf["rally_id"].isin(list(side_switches)).to_numpy().astype(int))
    elif switch_every:
        n_flips = total // switch_every - prior_points // switch_every
    else:
        n_flips = np.zeros(n, dtype=int)
    inverted = (n_flips % 2 == 1)
    # Equipo 1 = quien saca la primera jugada con lado conocido
    k0 = next((k for k, l in enumerate(rdf.serve_side) if l in (SIDE_LEFT, SIDE_RIGHT)), None)
    team1_side = SIDE_LEFT if k0 is None else rdf.serve_side.iloc[k0]
    if k0 is not None and inverted[k0]:
        team1_side = other_side(team1_side)                               # lado de Equipo 1 sin invertir
    team_left, team_right = [], []
    for inv in inverted:
        l1 = other_side(team1_side) if inv else team1_side
        team_left.append("Equipo 1" if l1 == SIDE_LEFT else "Equipo 2")
        team_right.append("Equipo 1" if l1 == SIDE_RIGHT else "Equipo 2")
    rdf["team_left"], rdf["team_right"] = team_left, team_right
    rdf["side_switch"] = np.r_[False, inverted[1:] != inverted[:-1]]
    rdf["serving_team"] = [(a if l == SIDE_LEFT else b) if isinstance(l, str) else None
                           for l, a, b in zip(rdf.serve_side, rdf.team_left, rdf.team_right)]
    return rdf


def _shots_with_result(info, rdf, fps, control_min_touches=2, control_min_sec=2.0):
    """
    Una fila por saque / ataque:
      - 'defendido' : el rival la recibió y la devolvió (hubo otro cruce después), o la
                      controló y la jugada siguió
      - 'punto'     : el golpe ganó el punto (cayó en la cancha rival o el rival no pudo
                      devolverla)
      - 'fuera'     : el golpe perdió el punto (afuera, a la red, o el saque no cruzó)
      - 'desconocido': no se pudo deducir el ganador
    Después del último cruce visto:
      - si el que recibió NO controló la pelota (0-1 toques y poco tiempo), el último
        golpe que cruzó es el que decidió el punto;
      - si la controló (varios toques o mucho tiempo), la jugada terminó con SU ataque:
        se agrega ese ataque (crossing_seen=False) con 'punto' si ganó ese equipo o 'fuera'
        si lo perdió (a la red / afuera), y el golpe anterior queda 'defendido'.
    opponent_touched distingue un punto directo (nadie la tocó) de uno donde el rival alcanzó a tocarla.

    Args:
        info (dict): {rally_id: {'possessions', 'crossings', 's0', 'e0'}} armado en compute_statistics.
        rdf (pd.DataFrame): jugadas con equipos y ganador asignados.
        fps (float): cuadros por segundo.
        control_min_touches (int): toques mínimos para considerar que el receptor controló.
        control_min_sec (float): tiempo mínimo de posesión para considerar que controló.

    Returns:
        pd.DataFrame: rally_id, frame, team, side, shot_type, result, opponent_touched, crossing_seen.
    """
    cols = ["rally_id", "frame", "team", "side", "shot_type", "result", "opponent_touched",
            "crossing_seen"]
    j = rdf.set_index("rally_id")
    rows_out = []
    for rid, inf in info.items():
        r = j.loc[rid]
        pos, crossings = inf["possessions"], inf["crossings"]
        win_side = r.winner_side if isinstance(r.winner_side, str) else None
        rally_shots = []
        for i, p in enumerate(pos[:-1]):
            if p["side"] is None:
                continue
            f_g = p["touches"][-1] if p["touches"] else crossings[i][0]
            rally_shots.append({"rally_id": rid, "frame": int(f_g), "side": p["side"],
                        "shot_type": "saque" if i == 0 else "ataque", "result": "defendido",
                        "opponent_touched": bool(pos[i + 1]["touches"]), "crossing_seen": True})
        last_pos = pos[-1]
        if len(pos) == 1:
            # el saque nunca se vio cruzar
            if last_pos["side"] is not None:
                controlled = False
                result_label = "desconocido" if win_side is None else ("punto" if win_side == last_pos["side"] else "fuera")
                rally_shots.append({"rally_id": rid, "frame": inf["s0"], "side": last_pos["side"], "shot_type": "saque",
                            "result": result_label, "opponent_touched": False, "crossing_seen": False})
        elif rally_shots:
            dur = (inf["e0"] - last_pos["start"]) / fps
            controlled = (last_pos["side"] is not None and len(last_pos["touches"]) > 0 and
                          (len(last_pos["touches"]) >= control_min_touches or dur >= control_min_sec))
            prev = rally_shots[-1]
            if win_side is None:
                prev["result"] = "desconocido"
            elif controlled:
                rally_shots.append({"rally_id": rid, "frame": int(last_pos["touches"][-1]), "side": last_pos["side"],
                            "shot_type": "ataque", "result": "punto" if win_side == last_pos["side"] else "fuera",
                            "opponent_touched": False, "crossing_seen": False})
            else:
                prev["result"] = "punto" if win_side == prev["side"] else "fuera"
        rows_out.extend(rally_shots)
    out = pd.DataFrame(rows_out, columns=[c for c in cols if c != "team"])
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["team"] = [j.loc[rid, "team_left"] if l == SIDE_LEFT else j.loc[rid, "team_right"]
                     for rid, l in zip(out.rally_id, out.side)]
    return out[cols].sort_values(["rally_id", "frame"]).reset_index(drop=True)


def _rallies_with_shots(rdf, shots):
    """
    Agrega a cada jugada el conteo de saques / ataques y cómo terminó.

    Args:
        rdf (pd.DataFrame): una fila por jugada.
        shots (pd.DataFrame): salida de _shots_with_result.

    Returns:
        pd.DataFrame: rdf con n_serves, serve_result, n_attacks, ataques por equipo,
            attacks_defended, ends_with y final_result (columnas principales primero).
    """
    rdf = rdf.copy()
    rows_out = []
    for rid in rdf.rally_id:
        g = shots[shots.rally_id == rid]
        s = g[g.shot_type == "saque"]
        a = g[g.shot_type == "ataque"]
        last_pos = g.iloc[-1] if len(g) else None
        rows_out.append({
            "rally_id": rid,
            "n_serves": len(s),
            "serve_result": s.result.iloc[0] if len(s) else None,
            "n_attacks": len(a),
            "attacks_team_1": int((a.team == "Equipo 1").sum()),
            "attacks_team_2": int((a.team == "Equipo 2").sum()),
            "attacks_defended": int((a.result == "defendido").sum()),
            "ends_with": None if last_pos is None else f"{last_pos.shot_type} {last_pos.team}",
            "final_result": None if last_pos is None else last_pos.result,
        })
    extra = pd.DataFrame(rows_out)
    rdf = rdf.merge(extra, on="rally_id", how="left")
    first_side = ["rally_id", "serve_frame", "end_frame", "serving_team", "n_serves", "serve_result",
               "n_attacks", "attacks_team_1", "attacks_team_2", "attacks_defended",
               "ends_with", "final_result", "winner_team", "winner_method"]
    return rdf[first_side + [c for c in rdf.columns if c not in first_side]]


def _team_summary(rdf, shots):
    """
    Resumen por equipo: puntos, saques, aces, ataques y eficacia.

    Args:
        rdf (pd.DataFrame): jugadas con winner_team.
        shots (pd.DataFrame): una fila por saque / ataque.

    Returns:
        pd.DataFrame: una fila por equipo.
    """
    rows_out = []
    for team in ["Equipo 1", "Equipo 2"]:
        g = shots[shots.team == team]
        s, a = g[g.shot_type == "saque"], g[g.shot_type == "ataque"]
        rival = shots[(shots.team != team) & shots.team.notna()]
        n_a = len(a)
        rows_out.append({
            "team": team,
            "points": int((rdf.winner_team == team).sum()),
            "serves": len(s), "aces": int((s.result == "punto").sum()),
            "serves_defended": int((s.result == "defendido").sum()),
            "serves_out": int((s.result == "fuera").sum()),
            "attacks": n_a,
            "attacks_won": int((a.result == "punto").sum()),
            "attacks_defended": int((a.result == "defendido").sum()),
            "attacks_out": int((a.result == "fuera").sum()),
            "attack_efficiency_pct": round(100 * (a.result == "punto").sum() / n_a, 1) if n_a else None,
            "opponent_attacks_defended": int(((rival.shot_type == "ataque") & (rival.result == "defendido")).sum()),
        })
    return pd.DataFrame(rows_out)
