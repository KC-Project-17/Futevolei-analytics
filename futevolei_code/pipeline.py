"""
Pipeline de futevôlei en DOS ETAPAS, con una revisión manual en el medio.

  ETAPA 1  stage1_detect(...)
      detecciones -> trayectoria -> repeticiones -> jugadas
      Genera el video de revisión (cada jugada numerada) y review.csv con el minuto
      de cada jugada en el video original y en el de revisión.

  REVISIÓN MANUAL (mirando review_video.mp4)
      - ¿Cada jugada es real?            -> anotar las falsas en `remove`
      - ¿Falta alguna?                   -> anotar inicio y fin (mm:ss del video original) en `add`
      - ¿En qué jugada cambian de lado?  -> anotar la PRIMERA jugada con los lados cambiados
                                            en `side_switches` (con la numeración de la etapa 1)
      - ¿Quién saca la jugada 1?         -> es "Equipo 1" (para `names`)

  ETAPA 2  stage2_statistics(...)
      aplica las correcciones -> movimiento de cámara -> homografías -> estadísticas
      Genera rally_stats / shots / summary y los videos finales (solo jugadas y con cajas).

Usa los demás módulos del paquete futevolei (ver futevolei/__init__.py).
Cada paso guarda su resultado en out_dir y lo reutiliza si ya existe (redo={...} para forzar).
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

from .ball_trajectory import build_ball_trajectory
from .broadcast import detect_replay_graphics, replay_mask_from_graphics
from .camera import camera_motion, net_from_pan
from .common import meta_to_dict
from .detection import remove_static_balls, remove_static_objects, track_play_events
from .match_stats import compute_statistics
from .rallies import detect_rallies
from .rally_homographies import compute_homographies_in_rallies
from .video_output import draw_detections, export_rallies


def _t(msg, t0):
    """
    Imprime un mensaje con el tiempo transcurrido desde t0.

    Args:
        msg (str): mensaje a imprimir.
        t0 (float): instante de inicio (time.time()).

    Returns:
        None
    """
    print(f"[{time.time() - t0:7.1f}s] {msg}")


def _mmss(seg):
    """
    Formatea segundos como 'm:ss.s'.

    Args:
        seg (float): segundos.

    Returns:
        str: tiempo formateado.
    """
    return f"{int(seg // 60)}:{seg % 60:04.1f}"


def _to_seconds(x):
    """
    Convierte 'mm:ss', 'mm:ss.s', 'hh:mm:ss' o un número de segundos a segundos.

    Args:
        x (str | int | float): tiempo a convertir.

    Returns:
        float: segundos.
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    parts = [float(p) for p in str(x).strip().split(":")]
    seg = 0.0
    for p in parts:
        seg = seg * 60 + p
    return seg


# Nombres de archivo de versiones anteriores (en español) -> nombres actuales
OLD_FILE_NAMES = {
    "rallies_etapa1.csv": "rallies_stage1.csv", "revision.csv": "review.csv",
    "video_revision.mp4": "review_video.mp4", "video_revision_boxes.mp4": "review_video_boxes.mp4",
    "movim.parquet": "cam_motion.parquet", "movim_firma.txt": "cam_motion_signature.txt",
    "homografias.pkl": "homographies.pkl", "homografias_firma.txt": "homographies_signature.txt",
    "jugadas.csv": "rally_stats.csv", "golpes.csv": "shots.csv", "resumen.csv": "summary.csv",
    "video_solo_jugadas.mp4": "rallies_video.mp4", "video_jugadas_boxes.mp4": "rallies_video_boxes.mp4",
}


def _migrate_old_files(out_dir):
    """
    Renombra los archivos guardados con los nombres de versiones anteriores, para que se
    sigan reutilizando (y no haya que recalcular el movimiento de cámara ni las homografías).

    Args:
        out_dir (str): carpeta de salida.

    Returns:
        list: nombres (viejos) de los archivos renombrados.
    """
    renamed = []
    for old, new in OLD_FILE_NAMES.items():
        a, b = os.path.join(out_dir, old), os.path.join(out_dir, new)
        if os.path.exists(a) and not os.path.exists(b):
            os.rename(a, b)
            renamed.append(old)
    if renamed:
        print(f"Archivos renombrados al formato nuevo: {renamed}")
    return renamed


def _review_table(rallies, meta):
    """
    Minuto de cada jugada en el video original y en el compilado de revisión.

    Args:
        rallies (pd.DataFrame): jugadas detectadas (detect_rallies).
        meta (pd.DataFrame): metadatos del video (fps, n_frames, ...).

    Returns:
        pd.DataFrame: una fila por jugada con rally_id, original_time,
            review_video_time, duration_sec, end_reason y columnas vacías para completar a mano.
    """
    fps = meta_to_dict(meta)["fps"]
    clip_dur = (rallies["clip_end"] - rallies["clip_start"] + 1) / fps
    comp_start = clip_dur.cumsum().shift(1).fillna(0.0)
    return pd.DataFrame({
        "rally_id": rallies["rally_id"],
        "original_time": [_mmss(s / fps) for s in rallies["serve_frame"]],
        "review_video_time": [_mmss(s) for s in comp_start],
        "duration_sec": rallies["duration_sec"],
        "end_reason": rallies["end_reason"],
        "is_correct": "",          # para completar a mano (si/no)
        "side_switch": "",       # marcar la primera jugada con los lados cambiados
    })


def stage1_detect(video_path, ball_player_model, out_dir="/content/salida",
                    redo=(), video_boxes=False, player_class_id=0, ball_class_id=1, verbose=True):
    """
    ETAPA 1: detecciones -> trayectoria -> repeticiones -> jugadas -> video de revisión.
    Cada paso se guarda en out_dir y se reutiliza si ya existe.

    Args:
        video_path (str): ruta del video original.
        ball_player_model: modelo YOLO de pelota y jugadores.
        out_dir (str): carpeta de salida.
        redo (iterable): pasos a recalcular aunque exista el archivo
            ("detections", "replays").
        video_boxes (bool): si True genera además el video de jugadas con las cajas
            (para ver dónde se pierde la pelota); tarda más.
        player_class_id (int): id de clase de los jugadores en el modelo.
        ball_class_id (int): id de clase de la pelota en el modelo.
        verbose (bool): imprime el progreso.

    Returns:
        dict: detections, detections_clean, meta, traj, replay_graphics, rallies,
            review (DataFrame), review_video (ruta) y review_video_boxes (ruta o None).
    """
    os.makedirs(out_dir, exist_ok=True)
    _migrate_old_files(out_dir)
    redo = set(redo)
    P = lambda name: os.path.join(out_dir, name)
    t0 = time.time()

    # 1. Detecciones (lo lento)
    if os.path.exists(P("detections.parquet")) and "detections" not in redo:
        detections = pd.read_parquet(P("detections.parquet"))
        meta = pd.read_csv(P("meta.csv"))
        _t("1. Detecciones cargadas de disco", t0)
    else:
        detections, meta = track_play_events(video_path, ball_player_model,
                                                player_class_id=player_class_id,
                                                ball_class_id=ball_class_id, verbose=verbose)
        detections.to_parquet(P("detections.parquet"))
        meta.to_csv(P("meta.csv"), index=False)
        _t("1. Detecciones calculadas", t0)

    # 2. Limpieza y trayectoria
    detections_clean = remove_static_balls(detections, meta)
    detections_clean = remove_static_objects(detections_clean, meta)
    traj = build_ball_trajectory(detections_clean, meta)
    _t("2. Limpieza y trayectoria", t0)

    # 3. Repeticiones por color
    if os.path.exists(P("replay_graphics.csv")) and "replays" not in redo:
        graphics = pd.read_csv(P("replay_graphics.csv"))
    else:
        graphics = detect_replay_graphics(video_path, verbose=verbose)
        graphics.to_csv(P("replay_graphics.csv"), index=False)
    in_replay = replay_mask_from_graphics(graphics, meta)
    _t(f"3. Repeticiones: {len(graphics)} animaciones", t0)

    # 4. Jugadas
    rallies = detect_rallies(traj, detections_clean, meta, replay_mask=in_replay)
    rallies.to_csv(P("rallies_stage1.csv"), index=False)
    _t(f"4. Jugadas: {len(rallies)}  {rallies['end_reason'].value_counts().to_dict()}", t0)

    # 5. Material de revisión
    review = _review_table(rallies, meta)
    review.to_csv(P("review.csv"), index=False)
    review_video = P("review_video.mp4")
    export_rallies(video_path, rallies, out_dir=None, compilation_path=review_video, draw_info=True)
    boxes_path = None
    if video_boxes:
        boxes_path = P("review_video_boxes.mp4")
        draw_detections(video_path, detections_clean, boxes_path, rallies=rallies,
                               traj=traj, verbose=False)
    _t("5. Video de revisión listo", t0)

    print("\nRevisá review_video.mp4 y completá para la etapa 2:"
          "\n  remove=[...]            jugadas falsas (numeración de este video)"
          "\n  add=[(ini, fin)]        jugadas que faltan, en mm:ss del video ORIGINAL"
          "\n  side_switches=[...]     primera jugada con los lados cambiados (numeración de este video,"
          "\n                          o 'mm:ss' del video original si es una jugada agregada)"
          "\n  names={...}             'Equipo 1' = quien saca la jugada 1")
    return {"detections": detections, "detections_clean": detections_clean, "meta": meta,
            "traj": traj, "replay_graphics": graphics, "rallies": rallies, "review": review,
            "review_video": review_video, "review_video_boxes": boxes_path}


def _apply_review(rallies, meta, remove=(), add=(), side_switches=None, margin_sec=1.0):
    """
    Aplica las correcciones de la revisión manual: elimina jugadas falsas,
    agrega las que faltan, renumera y traduce los cambios de lado a la numeración nueva.

    Args:
        rallies (pd.DataFrame): jugadas de la etapa 1.
        meta (pd.DataFrame): metadatos del video.
        remove (iterable): rally_id (numeración de la etapa 1) a eliminar.
        add (iterable): pares (inicio, fin) en 'mm:ss' del video original a agregar.
        side_switches (list | None): primera jugada con los lados cambiados, como id de la
            etapa 1 o 'mm:ss' del video original.
        margin_sec (float): margen en segundos antes/después de las jugadas agregadas.

    Returns:
        tuple: (rallies corregidas y renumeradas (pd.DataFrame),
            side_switches en la numeración nueva (list | None)).
    """
    fps = meta_to_dict(meta)["fps"]
    n = meta_to_dict(meta)["n_frames"]
    r = rallies[~rallies["rally_id"].isin(list(remove))].copy()
    r["origin"] = "detectada"
    r["stage1_id"] = r["rally_id"]
    new_rows = []
    for start_t, end_t in add:
        s0, e0 = int(_to_seconds(start_t) * fps), int(_to_seconds(end_t) * fps)
        new_rows.append({"serve_frame": s0, "end_frame": e0, "duration_sec": round((e0 - s0) / fps, 2),
                       "end_reason": "manual", "clip_start": max(0, s0 - int(margin_sec * fps)),
                       "clip_end": min(n - 1, e0 + int(margin_sec * fps)),
                       "serve_sec": round(s0 / fps, 2), "origin": "manual", "stage1_id": np.nan})
    if new_rows:
        r = pd.concat([r, pd.DataFrame(new_rows)], ignore_index=True)
    r = r.sort_values("serve_frame").reset_index(drop=True)
    r["rally_id"] = np.arange(1, len(r) + 1)
    # que los clips no se solapen
    prev_end = r["clip_end"].shift(1).fillna(-1).astype(int)
    r["clip_start"] = np.maximum(r["clip_start"], prev_end + 1)

    switches = None
    if side_switches is not None:
        switches = []
        for c in side_switches:
            if isinstance(c, str):                       # 'mm:ss' del video original
                f = _to_seconds(c) * fps
                k = r.index[r["serve_frame"] >= f - fps]
            else:                                        # id de la etapa 1
                k = r.index[r["stage1_id"] == c]
                if len(k) == 0:
                    raise ValueError(f"side_switches: la jugada {c} de la etapa 1 fue eliminada; "
                                     f"usá la siguiente o un 'mm:ss'")
            switches.append(int(r.at[k[0], "rally_id"]))
    return r, switches


def stage2_statistics(video_path, stage1, court_kp_model=None, out_dir="/content/salida",
                        remove=(), add=(), side_switches=None, names=None,
                        redo=(), video_boxes=True, rallies_only_video=False, verbose=True):
    """
    ETAPA 2: aplica las correcciones -> movimiento de cámara -> homografías ->
    estadísticas -> videos finales.

    Args:
        video_path (str): ruta del video original.
        stage1 (dict | None): el dict que devolvió stage1_detect (o None para cargar todo de out_dir).
        court_kp_model: modelo YOLO de pose de la cancha (None = sin homografías, menos preciso).
        out_dir (str): carpeta de salida (la misma de la etapa 1).
        remove (iterable): jugadas falsas a eliminar (numeración de la etapa 1).
        add (iterable): jugadas que faltan, pares (inicio, fin) en 'mm:ss' del video original.
        side_switches (list | None): primera jugada con los lados cambiados (numeración de la
            etapa 1 o 'mm:ss' del video original); None = cada 6 puntos.
        names (dict | None): renombra equipos, p.ej. {"Equipo 1": "...", "Equipo 2": "..."}.
        redo (iterable): pasos a recalcular aunque exista el archivo ("cam_motion", "homographies").
        video_boxes (bool): genera el video de jugadas con las cajas.
        rallies_only_video (bool): genera el video con solo las jugadas.
        verbose (bool): imprime el progreso.

    Returns:
        dict: rallies, side_switches, cam_motion, homographies, rally_stats, shots, summary,
            rallies_video (ruta o None) y video_boxes (ruta o None).
    """
    _migrate_old_files(out_dir)
    redo = set(redo)
    P = lambda name: os.path.join(out_dir, name)
    t0 = time.time()
    if stage1 is None:                                   # retomar otro día desde disco
        detections = pd.read_parquet(P("detections.parquet"))
        meta = pd.read_csv(P("meta.csv"))
        detections_clean = remove_static_objects(remove_static_balls(detections, meta), meta)
        traj = build_ball_trajectory(detections_clean, meta)
        rallies_stage1 = pd.read_csv(P("rallies_stage1.csv"))
    else:
        detections_clean, meta, traj = stage1["detections_clean"], stage1["meta"], stage1["traj"]
        rallies_stage1 = stage1["rallies"]

    # 1. Correcciones de la revisión
    rallies, switches = _apply_review(rallies_stage1, meta, remove, add, side_switches)
    rallies.to_csv(P("rallies.csv"), index=False)
    _t(f"1. Jugadas finales: {len(rallies)} ({len(list(remove))} eliminadas, "
       f"{len(list(add))} agregadas) | cambios de lado: {switches}", t0)

    # 2. Movimiento de cámara (se rehace si cambiaron las jugadas)
    signature = ",".join(map(str, rallies["serve_frame"].tolist()))
    cam_motion = None
    if os.path.exists(P("cam_motion.parquet")) and "cam_motion" not in redo and os.path.exists(P("cam_motion_signature.txt")):
        if open(P("cam_motion_signature.txt")).read() == signature:
            cam_motion = pd.read_parquet(P("cam_motion.parquet"))
            cam_motion = cam_motion.rename(columns={"corte": "cut"})   # caché de versiones anteriores
    if cam_motion is None:
        cam_motion = camera_motion(video_path, rallies, meta, verbose=verbose)
        cam_motion.to_parquet(P("cam_motion.parquet"))
        open(P("cam_motion_signature.txt"), "w").write(signature)
    _t("2. Movimiento de cámara", t0)

    # 3. Homografías (se rehacen si cambiaron las jugadas)
    homographies = {}
    cache_ok = (os.path.exists(P("homographies.pkl")) and os.path.exists(P("homographies_signature.txt"))
                and open(P("homographies_signature.txt")).read() == signature and "homographies" not in redo)
    if cache_ok:
        homographies = pickle.load(open(P("homographies.pkl"), "rb"))
        _t(f"3. Homografías cargadas: {len(homographies)}", t0)
    elif court_kp_model is not None:
        homographies = compute_homographies_in_rallies(video_path, court_kp_model, rallies, verbose=verbose)
        pickle.dump(homographies, open(P("homographies.pkl"), "wb"))
        open(P("homographies_signature.txt"), "w").write(signature)
        _t(f"3. Homografías calculadas: {len(homographies)}", t0)
    else:
        _t("3. Sin modelo de cancha: se sigue sin homografías (menos preciso)", t0)

    # 4. Estadísticas (ganador: caída con homografía si se ve; si no, quién saca la siguiente)
    stats = compute_statistics(rallies, traj, detections_clean, homographies, meta,
                                     cam_motion=cam_motion, side_switches=switches, verbose=verbose)
    rally_stats, shots, summary = stats["rally_stats"], stats["shots"], stats["summary"]
    if names:
        rally_stats, shots, summary = (df.replace(names) for df in (rally_stats, shots, summary))
    rally_stats.to_csv(P("rally_stats.csv"), index=False)
    shots.to_csv(P("shots.csv"), index=False)
    summary.to_csv(P("summary.csv"), index=False)
    _t("4. Estadísticas", t0)

    # 5. Videos finales
    rallies_video = boxes_path = None
    if rallies_only_video:
        rallies_video = P("rallies_video.mp4")
        export_rallies(video_path, rallies, out_dir=None, compilation_path=rallies_video, draw_info=True)
    if video_boxes:
        boxes_path = P("rallies_video_boxes.mp4")
        net, _ = net_from_pan(cam_motion, detections_clean, meta, homographies, verbose=False)
        draw_detections(video_path, detections_clean, boxes_path, rallies=rallies,
                               traj=traj, net_x=net, verbose=False)
    if rallies_video or boxes_path:
        _t("5. Videos listos", t0)

    return {"rallies": rallies, "side_switches": switches, "cam_motion": cam_motion, "homographies": homographies,
            "rally_stats": rally_stats, "shots": shots, "summary": summary,
            "rallies_video": rallies_video, "video_boxes": boxes_path}
