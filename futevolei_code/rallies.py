"""
Segmentación de jugadas de futevôlei.

Una jugada = desde que un jugador SACA (patea la pelota desde la arena)
hasta que la pelota TOCA LA ARENA o SALE del cuadro/cancha.
detect_rallies busca eventos de saque y de fin de jugada, y excluye solo las
repeticiones, los primeros planos y los cambios de cámara.

Unidades: todas las velocidades están en px/s y las distancias en px
NORMALIZADAS a 1080p (se escalan solas según la altura del video).
"""
import numpy as np
import pandas as pd

from .broadcast import replay_graphics_from_detections, replay_mask_from_graphics
from .common import find_runs, meta_to_dict


def _players_by_frame(df_det):
    """
    Cajas de los jugadores agrupadas por frame.

    Args:
        df_det (pd.DataFrame): detecciones.

    Returns:
        dict: {frame: array (k, 4) con x1, y1, x2, y2}.
    """
    p = df_det[df_det["type"] == "player"]
    return {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in p.groupby("frame")}


def _ball_near_feet(f, bx, by, players, search=5):
    """
    True si la pelota está en la zona baja (pies) de algún jugador cerca del frame f.

    Args:
        f (int): frame.
        bx (float): x de la pelota.
        by (float): y de la pelota.
        players (dict): salida de _players_by_frame.
        search (int): frames hacia atrás en los que se busca.

    Returns:
        bool
    """
    for g in range(f - search, f + 1):
        b = players.get(g)
        if b is None:
            continue
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        ok = ((bx >= b[:, 0] - 0.6 * bw) & (bx <= b[:, 2] + 0.6 * bw) &
              (by >= b[:, 1] + 0.5 * bh) & (by <= b[:, 3] + 0.3 * bh))
        if ok.any():
            return True
    return False


def _ball_in_upper_body(f, bx, by, players, search=3, tol=0.0):
    """
    True si la pelota está a la altura de cabeza/torso de un jugador.

    Args:
        f (int): frame.
        bx (float): x de la pelota.
        by (float): y de la pelota.
        players (dict): salida de _players_by_frame.
        search (int): frames antes y después en los que se busca.
        tol (float): amplía la caja (fracción del ancho/alto) para tolerar el borde del bbox.

    Returns:
        bool
    """
    for g in range(f - search, f + search + 1):
        b = players.get(g)
        if b is None:
            continue
        bw = b[:, 2] - b[:, 0]
        bh = b[:, 3] - b[:, 1]
        ok = ((bx >= b[:, 0] - tol * bw) & (bx <= b[:, 2] + tol * bw) &
              (by >= b[:, 1] - tol * bh) & (by <= b[:, 1] + 0.6 * bh))
        if ok.any():
            return True
    return False


def detect_rallies(
    traj,
    df_det,
    meta,
    # Continuidad
    join_gap_sec=2.0,          # huecos de pelota que se toleran dentro de una jugada (tapada en la red, etc.)
    join_gap_top_sec=3.0,      # hueco tolerado si la pelota salió por ARRIBA del cuadro (levantadas altas)
    # Saque
    v_rest=80,                 # px/s: pelota "quieta" antes del saque
    rest_sec=0.4,
    v_launch=350,              # px/s hacia arriba sostenidos = patada
    launch_frames=4,
    serve_check_sec=1.5,       # ventana para verificar que es un saque real
    serve_min_rise=150,        # px que debe subir la pelota (un saque va alto)
    serve_min_air_sec=0.8,     # el saque debe estar en el aire al menos esto (un pase a ras de arena no)
    serve_air_margin=80,       # px sobre el punto de patada para contar como "en el aire"
    serve_min_arc=0,           # (desactivado) px de arco mínimo del saque; falla con saques en primer plano + cambio de cámara
                               # Una pelota rodando que se aleja de la cámara sube en la imagen,
                               # pero en línea recta; un saque hace un arco (gravedad).
    serve_arc_sec=0.6,         # ventana tras la patada donde se mide el arco
    serve_min_dx=150,          # px que debe avanzar horizontalmente (cruzar hacia la red)
    require_player_at_serve=True,
    serve_filter=None,         # opcional: fn(frame, cx, cy) -> bool, p.ej. "detrás de la línea de fondo" vía homografía
    # Fin
    v_desc=150,                # px/s cayendo
    v_stop=90,                 # (no se usa: se mantiene por compatibilidad)
    stop_sec=0.25,
    contact_search_sec=0.5,
    ground_check_sec=0.4,      # tiempo tras el contacto en que la pelota NO debe volver a subir
    ground_rise_tol=30,        # px que puede "subir" en imagen rodando (perspectiva / cámara)
    v_touch_up=300,            # px/s hacia arriba: si sube así de rápido, fue un toque de jugador
    ground_recheck_sec=1.0,    # ventana larga tras el contacto para detectar defensas
    v_touch_up_late=420,       # px/s hacia arriba en esa ventana = la tocó un jugador (no era arena)
    ground_rest_sec=0.5,       # pelota quieta este tiempo = está en la arena (aunque no se haya visto caer)
    ground_rest_tol=30,               # px (1080p) que se puede mover "quieta" (ruido / cámara)
    ground_rest_min_cov=0.3,          # fracción mínima de frames con detección en la ventana de reposo
    ground_rest_max_jump=400,         # px: re-adquisición más lejos que esto = otra pelota (repuesto), no se acepta
    debug=False,               # imprime cada candidato a contacto con la arena y por qué se aceptó/rechazó
    edge_margin=0.03,          # fracción del ancho/alto para decir "salió del cuadro"
    # Filtros y márgenes
    replay_mask=None,          # frames de repetición (se ignoran). None = se detectan solos desde df_det
    auto_replay=True,          # detectar repeticiones con la animación visible en las detecciones
    closeup_h=0.40,            # jugador más alto > esto (fracción del alto) = primer plano, no es juego
    view_change_sec=0.3,       # tras un cambio primer plano <-> vista abierta, no se decide "ground"
    serve_rest_frac=0.25,      # fracción mínima de 1.5 s antes del saque con la pelota visible y quieta
    serve_after_replay=True,   # True: saque justo después de una repetición -> no exigir ver la pelota quieta
    ground_max_speed_after=1200,  # >0: si tras el "contacto" la pelota sigue a más de esto px/s, no es arena
    serve_rest_min_seen=3,     # >0: alternativa si la pelota está tapada: vista al menos esto frames y quieta en la mitad
    serve_wide_frac=0.5,       # fracción mínima de 1.5 s después del saque en vista abierta
    fall_min_det=3,            # detecciones reales mínimas durante una caída para creerla
    closeup_end_sec=0.5,       # un primer plano así de largo durante la jugada = la jugada terminó
    hidden_fall_sec=0.4,       # pelota que se pierde cayendo y no reaparece volando en este tiempo = ground
    point_cooldown_sec=5.0,    # tras terminar una jugada, ignorar "saques" durante este tiempo
                               # (el pase de la pelota al otro equipo ocurre 1-3 s después del punto;
                               #  el saque real viene después de celebración / repetición)
    min_rally_sec=1.0,         # una jugada real puede durar ~1.5 s (saque, recepción y cae)
    drop_caught_under_sec=3.0, # descarta "jugadas" cortas que terminan con la pelota atrapada (devoluciones al sacador)
    pre_roll_sec=1.0,
    post_roll_sec=1.0,
):
    """
    Busca las jugadas: desde el SAQUE hasta que la pelota toca la arena, la atrapan
    o sale del cuadro. Ignora repeticiones, primeros planos y cambios de cámara.
    Velocidades en px/s y distancias en px normalizados a 1080p.

    Args:
        traj (pd.DataFrame): trayectoria de la pelota (build_ball_trajectory).
        df_det (pd.DataFrame): detecciones limpias.
        meta (dict | pd.DataFrame): metadatos del video.
        join_gap_sec (float): huecos de pelota que se toleran dentro de una jugada
            (tapada en la red, etc.).
        join_gap_top_sec (float): hueco tolerado si la pelota salió por ARRIBA del cuadro
            (levantadas altas).
        v_rest (float): px/s: pelota "quieta" antes del saque.
        rest_sec (float): (no se usa: se mantiene por compatibilidad).
        v_launch (float): px/s hacia arriba sostenidos = patada.
        launch_frames (int): frames que debe sostenerse la subida.
        serve_check_sec (float): ventana para verificar que es un saque real.
        serve_min_rise (float): px que debe subir la pelota (un saque va alto).
        serve_min_air_sec (float): el saque debe estar en el aire al menos esto
            (un pase a ras de arena no).
        serve_air_margin (float): px sobre el punto de patada para contar como "en el aire".
        serve_min_arc (float): (desactivado con 0) px de arco mínimo del saque. Una pelota
            rodando que se aleja de la cámara sube en la imagen, pero en línea recta; un saque
            hace un arco (gravedad). Falla con saques en primer plano + cambio de cámara.
        serve_arc_sec (float): ventana tras la patada donde se mide el arco.
        serve_min_dx (float): px que debe avanzar horizontalmente (cruzar hacia la red).
        require_player_at_serve (bool): la pelota debe salir de los pies de un jugador.
        serve_filter (callable | None): opcional: fn(frame, cx, cy) -> bool, p. ej. "detrás de
            la línea de fondo" vía homografía.
        v_desc (float): px/s cayendo.
        v_stop (float): (no se usa: se mantiene por compatibilidad).
        stop_sec (float): (no se usa: se mantiene por compatibilidad).
        contact_search_sec (float): (no se usa: se mantiene por compatibilidad).
        ground_check_sec (float): tiempo tras el contacto en que la pelota NO debe volver a subir.
        ground_rise_tol (float): px que puede "subir" en imagen rodando (perspectiva / cámara).
        v_touch_up (float): px/s hacia arriba: si sube así de rápido, fue un toque de jugador.
        ground_recheck_sec (float): ventana larga tras el contacto para detectar defensas.
        v_touch_up_late (float): px/s hacia arriba en esa ventana = la tocó un jugador (no era arena).
        ground_rest_sec (float): pelota quieta este tiempo = está en la arena (aunque no se haya
            visto caer).
        ground_rest_tol (float): px (1080p) que se puede mover "quieta" (ruido / cámara).
        ground_rest_min_cov (float): fracción mínima de frames con detección en la ventana de reposo.
        ground_rest_max_jump (float): px: re-adquisición más lejos que esto = otra pelota
            (repuesto), no se acepta.
        debug (bool): imprime cada candidato a contacto con la arena y por qué se aceptó/rechazó.
        edge_margin (float): fracción del ancho/alto para decir "salió del cuadro".
        replay_mask (np.ndarray | None): frames de repetición (se ignoran). None = se detectan
            solos desde df_det.
        auto_replay (bool): detectar repeticiones con la animación visible en las detecciones.
        closeup_h (float): jugador más alto > esto (fracción del alto) = primer plano, no es juego.
        view_change_sec (float): tras un cambio primer plano <-> vista abierta, no se decide "ground".
        serve_rest_frac (float): fracción mínima de 1.5 s antes del saque con la pelota visible y quieta.
        serve_after_replay (bool): True: saque justo después de una repetición -> no exigir ver
            la pelota quieta.
        ground_max_speed_after (float): >0: si tras el "contacto" la pelota sigue a más de esto
            px/s, no es arena.
        serve_rest_min_seen (int): >0: alternativa si la pelota está tapada: vista al menos esto
            frames y quieta en la mitad.
        serve_wide_frac (float): fracción mínima de 1.5 s después del saque en vista abierta.
        fall_min_det (int): detecciones reales mínimas durante una caída para creerla.
        closeup_end_sec (float): un primer plano así de largo durante la jugada = la jugada terminó.
        hidden_fall_sec (float): pelota que se pierde cayendo y no reaparece volando en este
            tiempo = ground.
        point_cooldown_sec (float): tras terminar una jugada, ignorar "saques" durante este tiempo
            (el pase de la pelota al otro equipo ocurre 1-3 s después del punto; el saque real
            viene después de celebración / repetición).
        min_rally_sec (float): duración mínima de una jugada (una real puede durar ~1.5 s:
            saque, recepción y cae).
        drop_caught_under_sec (float): descarta "jugadas" cortas que terminan con la pelota
            atrapada (devoluciones al sacador).
        pre_roll_sec (float): segundos de margen antes del saque en el clip.
        post_roll_sec (float): segundos de margen después del fin en el clip.

    Returns:
        pd.DataFrame: una fila por jugada con rally_id, serve_frame, end_frame, duration_sec,
            end_reason ('ground', 'caught', 'out_of_frame', 'lost', 'lost_descending',
            'replay_cut', 'video_end'), clip_start, clip_end y serve_sec.
    """
    meta = meta_to_dict(meta)
    fps, n = meta["fps"], meta["n_frames"]
    W, H = meta["width"], meta["height"]
    s = H / 1080

    cx = traj["cx"].to_numpy()
    cy = traj["cy"].to_numpy()
    raw_cy = np.where(traj["detected"].to_numpy(), cy, np.nan)  # solo frames con detección real
    raw_cx = np.where(traj["detected"].to_numpy(), cx, np.nan)
    jump = traj["reacq_jump"].to_numpy() if "reacq_jump" in traj else np.full(len(cx), np.nan)
    grest_n = max(3, int(ground_rest_sec * fps))
    _b = df_det[df_det["type"] == "ball"].sort_values("frame")
    all_bf, all_bx, all_by = _b["frame"].to_numpy(), _b["cx"].to_numpy(), _b["cy"].to_numpy()

    def spot_had_ball(x, y, f0, f1, radius):
        """
        ¿Hubo detecciones de pelota en (x, y) entre f0 y f1? -> ahí hay una pelota de repuesto.

        Args:
            x (float): x del lugar.
            y (float): y del lugar.
            f0 (int): frame inicial.
            f1 (int): frame final (exclusivo).
            radius (float): radio de búsqueda (px).

        Returns:
            int: cantidad de detecciones de pelota en ese lugar.
        """
        a, b = np.searchsorted(all_bf, f0), np.searchsorted(all_bf, f1)
        if b <= a:
            return 0
        return int((np.hypot(all_bx[a:b] - x, all_by[a:b] - y) < radius).sum())
    check_n = int(ground_check_sec * fps)
    vy = traj["vy"].to_numpy()
    sp = traj["speed"].to_numpy()
    valid = ~np.isnan(cy)
    if replay_mask is None and auto_replay:
        _ev = replay_graphics_from_detections(df_det, meta)
        replay_mask = replay_mask_from_graphics(_ev, meta) if len(_ev) else None
    in_replay = np.zeros(n, dtype=bool) if replay_mask is None else np.asarray(replay_mask, dtype=bool)[:n]
    if in_replay.any():
        # La pelota de la repetición no existe para la segmentación
        cx = np.where(in_replay, np.nan, cx)
        cy = np.where(in_replay, np.nan, cy)
        vy = np.where(in_replay, np.nan, vy)
        sp = np.where(in_replay, np.nan, sp)
        raw_cx = np.where(in_replay, np.nan, raw_cx)
        raw_cy = np.where(in_replay, np.nan, raw_cy)
        valid = valid & ~in_replay
    # primer frame de repetición desde cada frame (n si no hay más)
    next_replay = np.full(n + 1, n)
    for f_ in range(n - 1, -1, -1):
        next_replay[f_] = f_ if in_replay[f_] else next_replay[f_ + 1]
    players = _players_by_frame(df_det)

    # Vista: primer plano (jugador muy grande) vs vista abierta del partido
    _p = df_det[df_det["type"] == "player"]
    _h = ((_p["y2"] - _p["y1"]) / H).groupby(_p["frame"]).max()
    hmax = _h.reindex(range(n)).to_numpy()
    hmax_s = pd.Series(hmax).rolling(max(1, int(0.3 * fps)), center=True, min_periods=1).median().to_numpy()
    closeup = np.nan_to_num(hmax_s, nan=0.0) > closeup_h
    flips = np.where(np.diff(closeup.astype(int)) != 0)[0] + 1
    near_change = np.zeros(n, dtype=bool)
    vc = int(view_change_sec * fps)
    for f_ in flips:
        near_change[max(0, f_ - vc // 2):min(n, f_ + vc)] = True
    no_ground = closeup | near_change   # aquí no se puede decidir que la pelota tocó la arena
    # Primeros planos largos (>= closeup_end_sec): la jugada ya terminó (celebración, jugador)
    long_closeup = np.zeros(n, dtype=bool)
    for a_, b_ in find_runs(closeup):
        if b_ - a_ >= closeup_end_sec * fps:
            long_closeup[a_:b_] = True

    chk_n = int(serve_check_sec * fps)
    min_rally_n = int(min_rally_sec * fps)

    # --- Bloques de presencia de la pelota, uniendo huecos cortos -----------
    blocks = []
    for a, b in find_runs(valid):
        if blocks:
            prev_end = blocks[-1][1] - 1
            exited_top = cy[prev_end] < 0.1 * H and vy[prev_end] < 0
            max_gap = (join_gap_top_sec if exited_top else join_gap_sec) * fps
            if a - prev_end <= max_gap:
                blocks[-1][1] = b
                continue
        blocks.append([a, b])

    def serve_at_cut(t):
        """
        Saque filmado en primer plano: la transmisión corta a la vista abierta justo
        cuando patean. Se reconoce por: primer plano -> vista abierta en t, pelota
        visible y quieta en el primer plano antes, y pelota volando después.

        Args:
            t (int): frame candidato.

        Returns:
            bool: True si en t hay un saque en el corte.
        """
        if t < 1 or not (closeup[t - 1] and not closeup[t]):
            return False
        # en el primer plano del saque la pelota a veces queda tapada el último segundo:
        # se mira una ventana más larga (3 s)
        w_pre = slice(max(0, t - int(3.0 * fps)), max(0, t - 3))
        quiet = ~np.isnan(raw_cy[w_pre]) & (np.nan_to_num(sp[w_pre], nan=1e9) < v_rest)
        if quiet.mean() < serve_rest_frac:
            return False
        w_post = slice(t, min(n, t + int(1.5 * fps)))
        if (~closeup[w_post]).mean() < serve_wide_frac:
            return False
        yy = raw_cy[w_post]
        if (~np.isnan(yy)).sum() < 10:
            return False
        rise = np.nanmax(yy) - np.nanmin(yy)
        flying = (np.nan_to_num(sp[w_post]) > v_flight_rest).any()
        return bool(flying and rise >= serve_min_rise * s)

    def find_serve(t0, t1, strict):
        """
        Primer frame en [t0, t1) que parece un saque.

        Args:
            t0 (int): frame inicial de búsqueda.
            t1 (int): frame final (exclusivo).
            strict (bool): (no se usa) reservado para un criterio más estricto.

        Returns:
            int | None: frame del saque, o None si no hay.
        """
        for t in range(t0, t1 - launch_frames):
            if serve_at_cut(t):
                if debug:
                    print(f"  saque en corte primer plano -> vista abierta t={t/fps:7.2f}s")
                return t
            # Patada: subida rápida y sostenida
            if not np.all(vy[t:t + launch_frames] < -v_launch):
                continue
            # Antes: pelota visible y quieta (en vista abierta o en el primer plano del saque).
            # Las repeticiones y los "saques" falsos no muestran ese reposo.
            w_pre = slice(max(0, t - int(1.5 * fps)), max(0, t - 3))
            seen = ~np.isnan(raw_cy[w_pre])
            quiet = seen & (np.nan_to_num(sp[w_pre], nan=1e9) < v_rest)
            # La pelota puede estar tapada por el sacador casi todo el tiempo antes del saque:
            # alcanza con que las pocas veces que se ve esté quieta
            quiet_ok = quiet.mean() >= serve_rest_frac or (
                serve_rest_min_seen > 0 and seen.sum() >= serve_rest_min_seen
                and quiet.sum() >= 0.5 * seen.sum())
            # Si antes del saque había una repetición (la transmisión vuelve justo al saque),
            # el reposo no se puede ver: no se exige
            if not quiet_ok and serve_after_replay and in_replay[w_pre].mean() >= 0.5:
                quiet_ok = True
            if not quiet_ok:
                continue
            # Después: la jugada se ve en vista abierta (no un primer plano / repetición)
            w_post = slice(t, min(n, t + int(1.5 * fps)))
            if (~closeup[w_post]).mean() < serve_wide_frac:
                if debug:
                    print(f"  saque descartado t={t/fps:7.2f}s: después sigue en primer plano")
                continue
            # Sale de los pies de un jugador
            if require_player_at_serve and not _ball_near_feet(t, cx[t], cy[t], players):
                continue
            # Trayectoria de saque: sube bastante y avanza hacia la red
            w = slice(t, min(t1, t + chk_n))
            rise = cy[t] - np.nanmin(cy[w])
            dx = np.nanmax(np.abs(cx[w] - cx[t]))
            if rise < serve_min_rise * s or dx < serve_min_dx * s:
                continue
            # Tiempo en el aire: hasta que vuelve a bajar cerca de la altura de la patada
            # (primero tiene que SUBIR sobre el margen; recién después se mide cuándo vuelve a bajar)
            seg = raw_cy[t:min(t1, t + chk_n)]
            thr = cy[t] - serve_air_margin * s
            up = np.where(~np.isnan(seg) & (seg < thr))[0]
            if len(up) == 0:
                air = 0.0
            else:
                after = seg[up[0]:]
                down = np.where(~np.isnan(after) & (after >= thr))[0]
                air = (up[0] + down[0]) / fps if len(down) else len(seg) / fps
            # Forma de arco: se ajusta una parábola y = c2*t² + c1*t + c0 a los primeros
            # serve_arc_sec tras la patada. En un saque c2 ≈ g/2 (curva hacia abajo);
            # rodando por la arena la trayectoria en imagen es casi recta (c2 ≈ 0).
            # "arco" = flecha de la parábola en esa ventana = c2 * T² / 4.
            wy = raw_cy[t:min(t1, t + int(serve_arc_sec * fps))]
            idx = np.where(~np.isnan(wy))[0]
            if len(idx) >= 6 and idx[-1] - idx[0] >= 0.5 * serve_arc_sec * fps:
                tt = idx / fps
                c2 = np.polyfit(tt, wy[idx], 2)[0]
                T = tt[-1] - tt[0]
                arc = max(0.0, c2) * T * T / 4 / s
            else:
                arc = serve_min_arc  # sin datos suficientes: no descartar por esto
            if arc < serve_min_arc:
                if debug:
                    print(f"  saque descartado t={t/fps:7.2f}s: trayectoria casi recta (arco {arc:.0f}px) -> pelota rodando / pase")
                continue
            if air < serve_min_air_sec:
                if debug:
                    print(f"  saque descartado t={t/fps:7.2f}s: solo {air:.2f}s en el aire (pase / patada baja)")
                continue
            if serve_filter is not None and not serve_filter(t, cx[t], cy[t]):
                continue
            return t
        return None

    def find_rest(start, limit):
        """
        Pelota en reposo sobre la arena: durante ground_rest_sec casi no se mueve.
        Cubre el caso en que la caída no se vio (tapada por un jugador) y la pelota
        aparece después quieta. El contacto se estima en la última caída previa.

        Args:
            start (int): frame del saque.
            limit (int): frame límite de búsqueda (exclusivo).

        Returns:
            int | None: frame estimado del contacto, o None.
        """
        t = start + int(0.4 * fps)
        min_det = max(3, int(ground_rest_min_cov * grest_n))
        last_live = start
        while t < limit - grest_n:
            if not np.isnan(sp[t]) and sp[t] > v_flight_rest:
                last_live = t
            if np.isnan(raw_cx[t]):
                t += 1
                continue
            wx, wy = raw_cx[t:t + grest_n], raw_cy[t:t + grest_n]
            ok = ~np.isnan(wx)
            px_, py_ = np.nanmedian(wx), np.nanmedian(wy)
            at_edge = (py_ < edge_margin * H or py_ > (1 - edge_margin) * H
                       or px_ < edge_margin * W or px_ > (1 - edge_margin) * W)
            if (ok.sum() >= min_det and np.ptp(wx[ok]) < ground_rest_tol * s
                    and np.ptp(wy[ok]) < ground_rest_tol * s and not no_ground[t] and not at_edge):
                # ¿Es la misma pelota que venía en juego? (no una de repuesto re-adquirida lejos)
                jumps = jump[last_live:t + 1]
                jumps = jumps[~np.isnan(jumps)]
                if len(jumps) and jumps.max() > ground_rest_max_jump:
                    if debug:
                        print(f"  reposo t={t/fps:7.2f}s descartado: re-adquisición a {jumps.max():.0f}px (otra pelota?)")
                    t += grest_n // 2
                    continue
                # ¿Había ya una pelota en ese lugar mientras se jugaba? -> repuesto, no la de juego
                px, py = np.nanmedian(wx), np.nanmedian(wy)
                prev = spot_had_ball(px, py, start, t - int(1.0 * fps), 40 * s)
                if prev >= 3:
                    if debug:
                        print(f"  reposo t={t/fps:7.2f}s descartado: ya había pelota ahí ({prev} detecciones) -> repuesto")
                    t += grest_n // 2
                    continue
                # Contacto = última caída en los 1.5 s previos, si la hay
                back = max(start, t - int(1.5 * fps))
                falling = np.where(vy[back:t + 1] > v_desc)[0]
                u = back + falling[-1] if len(falling) else t
                if debug:
                    print(f"  reposo t={t/fps:7.2f}s ({ok.sum()} detecciones) -> contacto estimado {u/fps:.2f}s")
                return u
            t += 1
        return None

    v_flight_rest = 300

    def find_end(start, limit):
        """
        Lo primero entre: caída sin rebote (find_end_fall) o pelota en reposo (find_rest).

        Args:
            start (int): frame del saque.
            limit (int): frame límite de búsqueda (exclusivo).

        Returns:
            tuple | None: (frame_fin, motivo 'ground' / 'caught'), o None si no se encontró.
        """
        a = find_end_fall(start, limit)
        b = find_rest(start, limit if a is None else a[0] + 1)
        if b is not None and (a is None or b < a[0]):
            reason = "caught" if _ball_in_upper_body(b, cx[b], cy[b], players) else "ground"
            return b, reason
        return a

    def find_end_fall(start, limit):
        """
        Contacto con la arena: la pelota viene cayendo, deja de caer y en los
        siguientes ground_check_sec NO vuelve a subir (puede quedarse quieta o rodar).
        Si vuelve a subir rápido, fue un toque de un jugador y la jugada sigue.

        Args:
            start (int): frame del saque.
            limit (int): frame límite de búsqueda (exclusivo).

        Returns:
            tuple | None: (frame_contacto, motivo 'ground' / 'caught'), o None.
        """
        t = start + int(0.4 * fps)  # saltar el despegue del saque
        while t < limit - launch_frames:
            if not np.all(vy[t:t + launch_frames] > v_desc):
                t += 1
                continue
            # Avanzar hasta donde termina la caída
            u = t + launch_frames
            while u < limit and not np.isnan(vy[u]) and vy[u] > 0.3 * v_desc:
                u += 1
            if u >= limit:
                break
            if np.isnan(vy[u]):
                # Se dejó de ver cayendo. Si no reaparece VOLANDO pronto, tocó la arena
                # (tapada por un jugador o fuera de cuadro abajo).
                last = u - 1
                while last > t and np.isnan(raw_cy[last]):
                    last -= 1
                nxt_seen = np.where(~np.isnan(raw_cy[u:min(limit, u + int(2.0 * fps))]))[0]
                fall_det = int((~np.isnan(raw_cy[t:u])).sum())
                if fall_det >= fall_min_det and not no_ground[last]:
                    if len(nxt_seen) == 0:
                        reappears_flying = False
                    else:
                        r0 = u + nxt_seen[0]
                        gap = r0 - last
                        ww = sp[r0:min(limit, r0 + int(0.3 * fps))]
                        vv = vy[r0:min(limit, r0 + int(0.3 * fps))]
                        flying = ((~np.isnan(ww)) & (ww > v_flight_rest)).any() or\
                                 ((~np.isnan(vv)) & (vv < -v_touch_up)).any()
                        reappears_flying = flying or gap < hidden_fall_sec * fps
                    if not reappears_flying:
                        if debug:
                            print(f"  caída perdida t={last/fps:7.2f}s: no reaparece volando -> GROUND")
                        return last, "ground"
                t = u + 1
                continue
            if (~np.isnan(raw_cy[t:u + 1])).sum() < fall_min_det or no_ground[u]:
                # caída inventada por interpolación, o cambio de cámara / primer plano
                t = u + 1
                continue
            w = slice(u, min(limit, u + check_n))
            det = ~np.isnan(raw_cy[w])
            if debug and det.sum() < 3:
                print(f"  candidato t={u/fps:7.2f}s  solo {det.sum()} detecciones después de caer -> sin decidir")
            if det.sum() >= 3:
                rise = cy[u] - np.nanmin(raw_cy[w])          # cuánto subió en imagen
                vyw = vy[w]
                max_up = -np.nanmin(vyw) if (~np.isnan(vyw)).any() else 0.0
                ok = rise < ground_rise_tol * s and max_up < v_touch_up
                if ok:
                    # Defensa con el cuerpo: la pelota frena a media altura, cae despacio y
                    # la vuelve a tocar otro jugador. Si en el segundo siguiente sube rápido,
                    # no tocó la arena.
                    wl = vy[u:min(limit, u + int(ground_recheck_sec * fps))]
                    late_up = -np.nanmin(wl) if (~np.isnan(wl)).any() else 0.0
                    if late_up > v_touch_up_late:
                        if debug:
                            print(f"  candidato t={u/fps:7.2f}s: vuelve a subir a {late_up:.0f}px/s "
                                  f"en {ground_recheck_sec}s -> defensa, sigue la jugada")
                        t = u + 1
                        continue
                if ok and ground_max_speed_after > 0:
                    # Una pelota que toca la arena se frena; si sigue volando rápido (pase bajo,
                    # recepción a ras) no es el final de la jugada
                    w_sp = sp[u + 1:min(limit, u + 1 + int(0.25 * fps))]
                    w_dt = ~np.isnan(raw_cy[u + 1:min(limit, u + 1 + int(0.25 * fps))])
                    if w_dt.sum() >= 3 and np.nanmedian(w_sp[w_dt]) > ground_max_speed_after:
                        if debug:
                            print(f"  candidato t={u/fps:7.2f}s: sigue a {np.nanmedian(w_sp[w_dt]):.0f}px/s -> no es arena")
                        t = u + 1
                        continue
                if ok and _ball_in_upper_body(u, cx[u], cy[u], players, tol=0.25):
                    # Se detuvo a la altura del cuerpo de un jugador: o la atrapó (fin)
                    # o fue un toque de pecho/cabeza y la pelota sigue cayendo (la jugada sigue)
                    w8 = raw_cy[u:min(limit, u + int(0.8 * fps))]
                    if (~np.isnan(w8)).any() and np.nanmax(w8) - cy[u] > 100 * s:
                        if debug:
                            print(f"  candidato t={u/fps:7.2f}s: toque en el cuerpo y sigue cayendo -> sigue la jugada")
                        t = u + 1
                        continue
                if debug:
                    print(f"  candidato t={u/fps:7.2f}s  detecciones={det.sum():2d}  "
                          f"subida={rise/s:5.0f}px (máx {ground_rise_tol})  "
                          f"vel_arriba={max_up:5.0f}px/s (máx {v_touch_up})  -> {'GROUND' if ok else 'toque'}")
                if ok:
                    reason = "caught" if _ball_in_upper_body(u, cx[u], cy[u], players) else "ground"
                    return u, reason
            t = u + 1
        return None

    def classify_loss(u):
        """
        La pelota dejó de verse sin un contacto claro.

        Args:
            u (int): último frame con pelota.

        Returns:
            str: 'out_of_frame', 'lost_descending' o 'lost'.
        """
        if cx[u] < edge_margin * W or cx[u] > (1 - edge_margin) * W or cy[u] > (1 - edge_margin) * H:
            return "out_of_frame"
        if vy[u] > v_desc:
            return "lost_descending"  # probablemente tocó la arena tapada por un jugador
        return "lost"

    rallies = []
    allowed_from = 0   # tras un punto, no se aceptan saques hasta este frame (vale entre bloques)
    for a, b in blocks:
        t = max(a, allowed_from)
        while t < b:
            start = find_serve(t, b, strict=False)
            if start is None:
                break
            nxt = find_serve(start + min_rally_n, b, strict=True)
            limit = nxt if nxt is not None else b

            result_label = find_end(start, limit)
            if result_label is not None:
                end, reason = result_label
            else:
                # Sin contacto claro: última pelota vista antes de un hueco largo
                idx = start + np.where(valid[start:max(limit, start + 1)])[0]
                gaps = np.where(np.diff(idx) > join_gap_sec * fps)[0]
                end = idx[gaps[0]] if len(gaps) else (idx[-1] if len(idx) else start)
                reason = classify_loss(end)

            # Si la transmisión pasa a un primer plano largo, la jugada terminó antes
            first_wide = start + int(np.argmax(~closeup[start:end + 1])) if (~closeup[start:end + 1]).any() else end
            lc = np.where(long_closeup[first_wide:end + 1])[0]
            if len(lc):
                cut_f = first_wide + lc[0]
                seen = np.where(valid[start:cut_f])[0]
                end = start + seen[-1] if len(seen) else cut_f - 1
                if reason not in ("ground", "caught"):
                    reason = classify_loss(end) if not np.isnan(cx[end]) else "lost"

            # Nunca pasar a la repetición
            if next_replay[start] <= end:
                end, reason = next_replay[start] - 1, "replay_cut"

            # Si la pelota se sigue viendo hasta el final del video, la jugada no terminó:
            # el video se corta en plena jugada
            if reason.startswith("lost") or reason == "out_of_frame":
                if end >= n - 1 - int(0.5 * fps):
                    reason = "video_end"

            dur = (end - start) / fps
            keep = end - start >= min_rally_n
            if reason == "caught" and dur < drop_caught_under_sec:
                keep = False
            if keep:
                rallies.append({"serve_frame": int(start), "end_frame": int(end),
                                "duration_sec": round(dur, 2), "end_reason": reason})
            if keep:
                allowed_from = end + 1 + int(point_cooldown_sec * fps)
            t = max(end + 1, allowed_from if keep else 0, start + 1)

    df = pd.DataFrame(rallies, columns=["serve_frame", "end_frame", "duration_sec", "end_reason"])
    if df.empty:
        return df

    # Márgenes para que el clip se vea natural, sin solaparse entre jugadas
    df["clip_start"] = (df["serve_frame"] - int(pre_roll_sec * fps)).clip(lower=0)
    df["clip_end"] = (df["end_frame"] + int(post_roll_sec * fps)).clip(upper=n - 1)
    # el margen final tampoco entra a la repetición
    df["clip_end"] = np.minimum(df["clip_end"], next_replay[df["end_frame"].to_numpy()] - 1)
    next_lc = np.full(n + 1, n)
    for f_ in range(n - 1, -1, -1):
        next_lc[f_] = f_ if long_closeup[f_] else next_lc[f_ + 1]
    df["clip_end"] = np.maximum(df["end_frame"], np.minimum(df["clip_end"], next_lc[df["end_frame"].to_numpy()] - 1))
    prev_end = df["clip_end"].shift(1).fillna(-1).astype(int)
    df["clip_start"] = np.maximum(df["clip_start"], prev_end + 1)
    df["serve_sec"] = (df["serve_frame"] / fps).round(2)
    df.insert(0, "rally_id", np.arange(1, len(df) + 1))
    return df
