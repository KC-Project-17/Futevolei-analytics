"""
Detección con YOLO y limpieza de detecciones.

    track_play_events      -> detecciones de pelota y jugadores de todo el video (una sola vez, guardar a disco)
    remove_static_balls    -> elimina pelotas de repuesto quietas en la arena
    remove_static_objects  -> elimina objetos fijos confundidos con la pelota (p. ej. en lo alto de la red)
"""
import cv2
import numpy as np
import pandas as pd

from .common import meta_to_dict


def track_play_events(
    video_path,
    model,
    player_class_id=0,
    ball_class_id=1,
    conf_ball=0.10,
    conf_player=0.35,
    imgsz=None,
    half=True,
    verbose=True,
):
    """
    Corre YOLO sobre todo el video en modo streaming.
    Usa un umbral de confianza por clase: bajo para la pelota (para no perderla)
    y más alto para jugadores (para evitar falsos positivos).

    Args:
        video_path (str): ruta del video.
        model: modelo YOLO de pelota y jugadores.
        player_class_id (int): id de clase de los jugadores.
        ball_class_id (int): id de clase de la pelota.
        conf_ball (float): confianza mínima para la pelota.
        conf_player (float): confianza mínima para los jugadores.
        imgsz (int | None): tamaño de inferencia. None = el tamaño con el que se entrenó el
            modelo (train_args del .pt). Conviene que coincida: la pelota ocupa pocos píxeles
            y con un imgsz menor se pierde.
        half (bool): inferencia en media precisión (GPU).
        verbose (bool): imprime el progreso.

    Returns:
        tuple: (detecciones, meta), ambos DataFrames.
            detecciones: frame, type ('ball' / 'player'), x1, y1, x2, y2, cx, cy, conf.
            meta: una fila con fps, width, height, n_frames.
    """
    if imgsz is None:
        try:
            imgsz = model.ckpt["train_args"]["imgsz"]
        except Exception:
            imgsz = model.overrides.get("imgsz", 1280)
    if verbose:
        print(f"imgsz de inferencia: {imgsz}")
    cap = cv2.VideoCapture(video_path)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    if verbose:
        print(f"Analizando video: {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps, "
              f"{meta['n_frames']} frames")

    rows = []
    frame_idx = -1
    results = model.predict(
        source=video_path,
        stream=True,
        conf=min(conf_ball, conf_player),
        imgsz=imgsz,
        half=half,
        verbose=False,
    )
    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            clss = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for c, (x1, y1, x2, y2), s in zip(clss, xyxy, confs):
                if c == ball_class_id and s >= conf_ball:
                    obj_type = "ball"
                elif c == player_class_id and s >= conf_player:
                    obj_type = "player"
                else:
                    continue
                rows.append((frame_idx, obj_type, x1, y1, x2, y2,
                             (x1 + x2) / 2, (y1 + y2) / 2, s))

        if verbose and frame_idx % 3000 == 0:
            print(f"Procesando frame {frame_idx}/{meta['n_frames']}")

    df = pd.DataFrame(rows, columns=["frame", "type", "x1", "y1", "x2", "y2", "cx", "cy", "conf"])
    # CAP_PROP_FRAME_COUNT a veces no es exacto
    meta["n_frames"] = max(meta["n_frames"], frame_idx + 1)
    return df, pd.DataFrame([meta])       # meta como DataFrame de una fila (se guarda con .to_csv)


def remove_static_balls(df_det, meta, cell_px=20, window_sec=8.0, min_coverage=0.4):
    """
    Elimina detecciones de pelota que están quietas en el mismo lugar durante
    mucho tiempo ANTES y DESPUÉS del frame (pelotas de repuesto).
    La pelota del saque se conserva: está quieta antes, pero no después (la patean).
    Ojo: con paneos de cámara fuertes el filtro es menos efectivo (las pelotas
    quietas se "mueven" en píxeles), pero en ese caso tampoco elimina de más.

    Args:
        df_det (pd.DataFrame): detecciones (track_play_events).
        meta (dict | pd.DataFrame): metadatos del video.
        cell_px (float): tamaño de la celda de la grilla (px a 1080p).
        window_sec (float): ventana (s) antes y después del frame.
        min_coverage (float): fracción mínima de frames con pelota en la celda (antes y después).

    Returns:
        pd.DataFrame: df_det sin las pelotas estáticas.
    """
    meta = meta_to_dict(meta)
    fps = meta["fps"]
    scale = meta["height"] / 1080
    cell = cell_px * scale
    win = int(window_sec * fps)

    balls = df_det[df_det["type"] == "ball"]
    if balls.empty:
        return df_det

    gx = (balls["cx"].to_numpy() // cell).astype(int)
    gy = (balls["cy"].to_numpy() // cell).astype(int)
    frames = balls["frame"].to_numpy()

    # Frames con detección por celda
    cells = {}
    for f, x, y in zip(frames, gx, gy):
        cells.setdefault((x, y), []).append(f)
    cells = {k: np.unique(v) for k, v in cells.items()}

    neigh_cache = {}
    is_static = np.zeros(len(balls), dtype=bool)
    for i, (f, x, y) in enumerate(zip(frames, gx, gy)):
        key = (x, y)
        if key not in neigh_cache:
            arrs = [cells[(x + dx, y + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (x + dx, y + dy) in cells]
            neigh_cache[key] = np.unique(np.concatenate(arrs))
        fr = neigh_cache[key]
        past = np.searchsorted(fr, f) - np.searchsorted(fr, f - win)
        future = np.searchsorted(fr, f + win, side="right") - np.searchsorted(fr, f, side="right")
        is_static[i] = past >= min_coverage * win and future >= min_coverage * win

    drop_idx = balls.index[is_static]
    print(f"Pelotas estáticas eliminadas: {len(drop_idx)} de {len(balls)} detecciones")
    return df_det.drop(index=drop_idx)


def remove_static_objects(df_det, meta, link_px=25, max_gap_frames=10, still_px=20,
                          still_vx=200, other_min_dist=150, moving_ptp=60,
                          track_concurrent=0.2, band_px=20, band_min_sec=10.0,
                          band_concurrent=0.1, verbose=True):
    """
    Elimina objetos fijos que el modelo confunde con la pelota (p. ej. algo en lo alto
    de la red). La cámara casi solo panea en horizontal, así que esos objetos quedan a
    altura constante (cy) aunque se muevan en x.

    Un "track quieto" (casi sin movimiento vertical y con poca velocidad horizontal) es
    falso si, mientras existe, hay OTRA pelota moviéndose lejos: la pelota real no puede
    estar en dos lugares. Además, si una franja de altura acumula muchos tracks quietos
    falsos, se eliminan todos los tracks quietos de esa franja (también cuando la pelota
    real no se ve, que es justo cuando confundían al seguimiento).
    La pelota esperando el saque nunca coincide con otra pelota en movimiento, así que
    no se toca.

    Args:
        df_det (pd.DataFrame): detecciones.
        meta (dict | pd.DataFrame): metadatos del video.
        link_px (float): distancia máxima (px a 1080p) para unir detecciones en un track.
        max_gap_frames (int): frames sin detección que tolera un track.
        still_px (float): movimiento vertical máximo de un track quieto.
        still_vx (float): velocidad horizontal máxima (px/s) de un track quieto.
        other_min_dist (float): distancia mínima a la otra pelota para contarla como "lejos".
        moving_ptp (float): recorrido vertical mínimo de un track para considerarlo en movimiento.
        track_concurrent (float): fracción del track con otra pelota moviéndose lejos para descartarlo.
        band_px (float): alto de las franjas de altura.
        band_min_sec (float): segundos mínimos de tracks quietos en una franja.
        band_concurrent (float): fracción mínima de concurrencia en la franja.
        verbose (bool): imprime el resumen.

    Returns:
        pd.DataFrame: df_det sin esas detecciones.
    """
    meta = meta_to_dict(meta)
    fps = meta["fps"]
    s = meta["height"] / 1080
    balls = df_det[df_det["type"] == "ball"].sort_values("frame")
    if balls.empty:
        return df_det
    fr = balls["frame"].to_numpy()
    xs = balls["cx"].to_numpy()
    ys = balls["cy"].to_numpy()

    # 1) tracks cortos por cercanía
    tid = np.full(len(balls), -1)
    active, ntr = [], 0
    for i in range(len(balls)):
        f, x, y = fr[i], xs[i], ys[i]
        active = [a for a in active if f - a[1] <= max_gap_frames]
        best, bd = None, link_px * s
        for a in active:
            if a[1] == f:
                continue
            dd = np.hypot(a[2] - x, a[3] - y)
            if dd < bd:
                best, bd = a, dd
        if best is None:
            best = [ntr, f, x, y]
            active.append(best)
            ntr += 1
        else:
            best[1], best[2], best[3] = f, x, y
        tid[i] = best[0]
    t = pd.DataFrame({"idx": balls.index.to_numpy(), "frame": fr, "cx": xs, "cy": ys, "tid": tid})
    g = t.groupby("tid")
    info = pd.DataFrame({"f0": g.frame.min(), "f1": g.frame.max(), "n": g.size(),
                         "ptp_y": g.cy.max() - g.cy.min(), "ptp_x": g.cx.max() - g.cx.min(),
                         "cy": g.cy.median()})
    info["dur"] = (info.f1 - info.f0) / fps
    info["still"] = (info.ptp_y < still_px * s) & (info.ptp_x / info.dur.clip(lower=1 / fps) < still_vx * s)
    moving = info.index[info.ptp_y > moving_ptp * s]

    # 2) concurrencia con una pelota en movimiento lejos
    tm = t[t.tid.isin(moving)]
    mf = {f: grp[["cx", "cy"]].to_numpy() for f, grp in tm.groupby("frame")}
    cand = info[info.still & (info.dur >= 0.5)]
    conc = {}
    for k in cand.index:
        pts = t[t.tid == k]
        c = 0
        for f, x, y in zip(pts.frame, pts.cx, pts.cy):
            m = mf.get(f)
            if m is not None and (np.hypot(m[:, 0] - x, m[:, 1] - y) > other_min_dist * s).any():
                c += 1
        conc[k] = c / len(pts)
    cand = cand.assign(conc=pd.Series(conc))
    fake_tracks = set(cand.index[cand.conc >= track_concurrent])

    # 3) franjas de altura con muchos tracks quietos falsos
    fake_bands = []
    for y0 in np.arange(0, meta["height"], band_px * s):
        inb = cand[(cand.cy >= y0) & (cand.cy < y0 + band_px * s)]
        if inb.empty:
            continue
        secs = inb.n.sum() / fps
        cfrac = (inb.conc * inb.n).sum() / inb.n.sum()
        if secs >= band_min_sec and cfrac >= band_concurrent:
            fake_bands.append(float(np.median(inb.cy)))
    for yb in fake_bands:
        fake_tracks |= set(info.index[info.still & ((info.cy - yb).abs() < band_px * s)])

    drop = t[t.tid.isin(fake_tracks)]["idx"].tolist()
    if verbose:
        bands = ", ".join(f"y≈{y:.0f}px" for y in fake_bands) or "ninguna"
        print(f"Objetos fijos confundidos con la pelota: {len(drop)} detecciones eliminadas "
              f"(franjas: {bands})")
    return df_det.drop(index=drop)
