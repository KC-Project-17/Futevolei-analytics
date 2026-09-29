"""
Homografías aplicadas a las jugadas:
    compute_homographies_in_rallies -> homografías solo en los frames de las jugadas
                                       (mismo algoritmo que homographies.compute_homographies)
    stabilize_homographies          -> una homografía robusta para cada frame, usando el paneo
                                       de la cámara (camera.camera_motion)
"""
import cv2
import numpy as np

from . import homographies as hg


def compute_homographies_in_rallies(video_path, court_kp_model, rallies, conf_thresh=0.35,
                                    step=2, verbose=True):
    """
    Igual que compute_homographies() de homographies.py, pero corriendo el modelo de
    keypoints solo en los frames de cada jugada (clip_start..clip_end) y 1 de cada
    `step` frames. En un partido son ~5 min de 34 -> ~7x menos trabajo (x2 con step=2).
    compute_statistics usa la H más cercana (<= 15 frames).

    Args:
        video_path (str): ruta del video.
        court_kp_model: modelo YOLO de pose de la cancha.
        rallies (pd.DataFrame): jugadas (clip_start, clip_end).
        conf_thresh (float): confianza mínima para considerar un keypoint visible.
        step (int): se procesa 1 de cada `step` frames.
        verbose (bool): imprime el progreso.

    Returns:
        dict: {frame: H} imagen -> cancha 1800x900.
    """
    cap = cv2.VideoCapture(video_path)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    segments = [(int(a), int(b)) for a, b in zip(rallies.clip_start, rallies.clip_end)]
    targets = set()
    for a, b in segments:
        targets.update(range(a, b + 1, step))

    # Pasada 1: keypoints (lectura secuencial; los frames fuera de jugada solo se saltan)
    kpts_per_frame = {}
    last_needed = max(b for _, b in segments)
    f = 0
    while f <= last_needed:
        if f in targets:
            ok, frame = cap.read()
            if not ok:
                break
            kpts, confs = hg.predict_court_keypoints(court_kp_model, frame, input_size=1280, conf=0.25, iou=0.5)
            if kpts is not None:
                kpts_per_frame[f] = (kpts, confs)
        else:
            if not cap.grab():
                break
        if verbose and f % 20000 == 0:
            print(f"Homografías: frame {f}/{last_needed}")
        f += 1
    cap.release()

    # Homografías pooled con todos los frames de todas las jugadas
    pooled = hg.compute_pooled_homographies(kpts_per_frame, img_w, img_h, conf_thresh)

    # Pasada 2: hacia adelante, arrastrando H_last de una jugada a la siguiente como el
    # pipeline original (si una jugada nunca muestra 4 puntos, el Caso B necesita una H previa;
    # el chequeo de plausibilidad descarta las que no calzan)
    homs = {}
    H_last = None
    for a, b in segments:
        for f in range(a, b + 1, step):
            if f in kpts_per_frame:
                kpts, confs = kpts_per_frame[f]
                vis = hg.visible_keypoints(confs, conf_thresh)
                H_pool = hg.find_compatible_pooled_homography(vis, pooled)
                H, _ = hg.get_footvolley_homography(kpts, confs, img_w=img_w, img_h=img_h,
                                                       conf_thresh=conf_thresh,
                                                       H_last=H_pool if H_pool is not None else H_last)
                if H is not None:
                    H_last = H
                    homs[f] = H
    # Pasada 3: hacia atrás, igual que el bootstrap de compute_homographies(). Sin esto las
    # primeras jugadas (antes de la primera H lograda) nunca tienen referencia para el Caso B
    # y quedan sin ninguna homografía. Solo se completan frames que no tienen H.
    H_next = None
    for a, b in reversed(segments):
        own = [f for f in range(a, b + 1, step) if f in homs]
        if own:
            H_next = homs[max(own)]
        for f in range(b - (b - a) % step, a - 1, -step):
            if f in homs:
                H_next = homs[f]
                continue
            if f in kpts_per_frame and H_next is not None:
                kpts, confs = kpts_per_frame[f]
                vis = hg.visible_keypoints(confs, conf_thresh)
                H_pool = hg.find_compatible_pooled_homography(vis, pooled)
                H, _ = hg.get_footvolley_homography(kpts, confs, img_w=img_w, img_h=img_h,
                                                       conf_thresh=conf_thresh,
                                                       H_last=H_pool if H_pool is not None else H_next)
                if H is not None:
                    H_next = H
                    homs[f] = H
    if verbose:
        no_info = [i for i, (a, b) in enumerate(segments) if not any(a <= f <= b for f in homs)]
        if no_info:
            print(f"⚠️  jugadas sin ninguna homografía (índices): {no_info}")
        print(f"Homografías calculadas: {len(homs)} frames en {len(segments)} jugadas "
              f"({len(pooled)} planos pooled)")
    return homs


def stabilize_homographies(homographies, cam_motion, window_sec=1.0, min_h=5, max_dev_px=60.0,
                            fps=60.0, verbose=True):
    """
    Homografía para CADA frame de las jugadas, robusta a homografías sueltas malas.
    Dentro de un plano la cámara solo gira: una H del frame s sirve para el frame t
    corriendo la imagen lo que giró la cámara entre s y t (dx acumulado de cam_motion).
    Para cada frame t se toman todas las H del mismo plano a <= window_sec, se llevan
    al frame t, y se combinan: se proyectan las 6 marcas de la cancha con cada una, se
    toma la MEDIANA de cada marca (descarta las H malas) y se ajusta una H con esas
    medianas. Las H que se alejan más de max_dev_px de la mediana se descartan.

    Args:
        homographies (dict): {frame: H} imagen -> cancha.
        cam_motion (pd.DataFrame): salida de camera_motion.
        window_sec (float): ventana (s) alrededor de cada frame.
        min_h (int): homografías mínimas en el plano / ventana.
        max_dev_px (float): desvío máximo (px) respecto de la mediana para no descartar una H.
        fps (float): cuadros por segundo del video.
        verbose (bool): imprime el resumen.

    Returns:
        dict: {frame: H} (imagen -> cancha 1800x900), como homographies.
    """
    landmarks = np.array([[0, 0], [900, 0], [1800, 0], [1800, 900], [900, 900], [0, 900]], np.float32)
    mv = cam_motion.sort_values("frame").reset_index(drop=True)
    fr = mv["frame"].to_numpy()
    is_new = mv["cut"].to_numpy().astype(bool) | (np.diff(fr, prepend=-10) != 1)
    shot_id = np.cumsum(is_new)
    w = int(window_sec * fps)
    out, n_desc, n_tot = {}, 0, 0
    for pid in np.unique(shot_id):
        idx = np.where(shot_id == pid)[0]
        f_pl = fr[idx]
        dx = mv["dx"].to_numpy()[idx].copy()
        dx[0] = 0.0
        c = np.cumsum(dx)
        cpos = dict(zip(f_pl, c))
        with_h = [f for f in f_pl if f in homographies]
        if len(with_h) < min_h:
            continue
        # marcas de cancha en la imagen de cada frame con H, pasadas a "coordenadas del plano"
        # (restando el giro acumulado): así todas son comparables
        img_pl = {}
        for f in with_h:
            try:
                pts = cv2.perspectiveTransform(landmarks.reshape(-1, 1, 2), np.linalg.inv(homographies[f])).reshape(-1, 2)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(pts)):
                continue
            pts[:, 0] -= cpos[f]
            img_pl[f] = pts
        fh = np.array(sorted(img_pl))
        if len(fh) < min_h:
            continue
        P = np.stack([img_pl[f] for f in fh])                       # (n, 6, 2)
        for f in f_pl:
            sel = np.abs(fh - f) <= w
            if sel.sum() < min_h:
                sel = np.ones(len(fh), bool)                          # todo el plano
            Q = P[sel]
            med = np.median(Q, axis=0)
            dev = np.median(np.linalg.norm(Q - med, axis=2), axis=1)  # desvío típico de cada H
            ok = dev <= max_dev_px
            n_tot += int(sel.sum()); n_desc += int((~ok).sum())
            if ok.sum() >= max(2, min_h // 2):
                med = np.median(Q[ok], axis=0)
            src = med.copy()
            src[:, 0] += cpos[f]                                      # de vuelta a la imagen del frame f
            H, _ = cv2.findHomography(src.astype(np.float32), landmarks, 0)
            if H is not None:
                out[int(f)] = H
    if verbose:
        print(f"Homografías estabilizadas: {len(out)} frames (antes {len(homographies)}); "
              f"descartadas por inconsistentes ~{100 * n_desc / max(1, n_tot):.0f}% de las usadas")
    return out
