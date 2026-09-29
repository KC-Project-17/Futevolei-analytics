"""
Cámara y red. La transmisión usa UNA cámara al costado de la red que GIRA siguiendo la
pelota: la red se mueve en la imagen y muchas veces queda fuera de cuadro. Por eso se mide
el paneo de la cámara (camera_motion) y con él se sigue la red en cada frame (net_from_pan).
"""
import cv2
import numpy as np
import pandas as pd

from .common import meta_to_dict
from .court_geometry import net_x_in_image


def camera_motion(video_path, rallies, meta, margin_sec=0.5, small_w=480,
                      hist_thr=0.25, hist_thr_low=0.12, resp_cut=0.10, verbose=True):
    """
    Lee el video SOLO dentro de cada jugada y mide, frame a frame:
      - dx: cuántos px (en 1920 de ancho) se corrió la imagen por el paneo de la
            cámara (phaseCorrelate sobre la franja de gradas, que tiene mucha textura
            y no tiene gráficos encima). dx > 0 = el contenido se mueve a la derecha.
      - cut: cambio de plano (el histograma de color cambia de golpe). En el primer
            frame de cada jugada también se marca corte.
    ~1-2 min para un partido completo (solo lee los frames de las jugadas).

    Args:
        video_path (str): ruta del video.
        rallies (pd.DataFrame): jugadas (serve_frame, end_frame).
        meta (dict | pd.DataFrame): metadatos del video.
        margin_sec (float): segundos extra antes y después de cada jugada.
        small_w (int): ancho al que se reduce el frame para medir.
        hist_thr (float): diferencia de histograma que por sí sola indica corte.
        hist_thr_low (float): diferencia de histograma que indica corte si además el ajuste es malo.
        resp_cut (float): respuesta de phaseCorrelate por debajo de la cual el ajuste es malo.
        verbose (bool): imprime el progreso.

    Returns:
        pd.DataFrame: frame, dx, resp (calidad del ajuste 0-1), hd (diferencia de histograma), cut.
    """
    meta = meta_to_dict(meta)
    fps, W, Hh, n = meta["fps"], meta["width"], meta["height"], meta["n_frames"]
    small_h = int(round(small_w * Hh / W))
    scale_factor = W / small_w
    y0, y1 = int(0.15 * small_h), int(0.74 * small_h)          # gradas + fondo de la cancha
    m = int(margin_sec * fps)
    cap = cv2.VideoCapture(video_path)
    rows_out = []
    ya = set()
    for k, (_, r) in enumerate(rallies.iterrows()):
        a = max(0, int(r["serve_frame"]) - m)
        b = min(n - 1, int(r["end_frame"]) + m)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        prev, win = None, None
        for f in range(a, b + 1):
            ok, im = cap.read()
            if not ok:
                break
            small = cv2.resize(im, (small_w, small_h), interpolation=cv2.INTER_AREA)
            band_img = cv2.cvtColor(small[y0:y1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            if win is None:
                win = cv2.createHanningWindow((band_img.shape[1], band_img.shape[0]), cv2.CV_32F)
            hist = cv2.calcHist([cv2.resize(small, (160, 90))], [0, 1, 2], None, [8, 8, 8],
                                [0, 256] * 3).ravel()
            hist /= hist.sum() + 1e-9
            if prev is None:
                dx, resp, hd, is_cut = 0.0, 1.0, 0.0, True
            else:
                (dx, _), resp = cv2.phaseCorrelate(prev[0], band_img, win)
                hd = 0.5 * float(np.abs(hist - prev[1]).sum())
                is_cut = hd > hist_thr or (hd > hist_thr_low and resp < resp_cut)
                dx = 0.0 if is_cut else float(dx) * scale_factor
            prev = (band_img, hist)
            if f not in ya:
                rows_out.append((f, dx, float(resp), hd, bool(is_cut)))
                ya.add(f)
        if verbose and (k + 1) % 10 == 0:
            print(f"Cámara: {k + 1}/{len(rallies)} jugadas")
    cap.release()
    mv = pd.DataFrame(rows_out, columns=["frame", "dx", "resp", "hd", "cut"])
    if verbose:
        print(f"Cámara: {len(mv)} frames leídos, {int(mv.cut.sum()) - len(rallies)} cortes dentro de jugadas")
    return mv


def net_from_pan(cam_motion, df_det, meta, homographies=None, closeup_h=0.40, min_frames_h=10,
                  min_intervals=10, verbose=True):
    """
    Posición x de la red en CADA frame de las jugadas, aunque la red no se vea.
    Dentro de un plano (entre cortes) la cámara solo gira, así que la red está en
    x = N + c(t), con c(t) = paneo acumulado. N (la red "en coordenadas del plano")
    se estima una vez por plano con:
      1) homografía: mediana de la red proyectada (si el plano tiene >= min_frames_h
         frames con H). Es la más precisa (~±40 px).
      2) jugadores: nunca cruzan la red. Con el paneo descontado, en los frames con 4
         jugadores separados 2|2 la red está entre la 2ª y la 3ª posición; se busca el
         valor que cumple eso en más frames y, dentro de esa zona, el hueco sin jugadores
         (~±100 px).
      3) si hay pocos frames con 4 jugadores: el hueco más ancho sin jugadores (~±200 px).

    Args:
        cam_motion (pd.DataFrame): salida de camera_motion.
        df_det (pd.DataFrame): detecciones limpias (pelota + jugadores).
        meta (dict | pd.DataFrame): metadatos del video.
        homographies (dict | None): {frame: H} imagen -> cancha.
        closeup_h (float): altura relativa de jugador a partir de la cual es primer plano (se ignora).
        min_frames_h (int): frames con H mínimos en un plano para usar la homografía.
        min_intervals (int): frames 2|2 mínimos para usar la regla de los jugadores.
        verbose (bool): imprime el resumen.

    Returns:
        tuple: (net_x, info). net_x = array de largo n_frames (NaN fuera de jugadas o en
            planos sin referencia, p. ej. el primer plano del sacador); info = DataFrame con
            una fila por plano (cam_shot, start, end, source, n_ref, N).
    """
    meta = meta_to_dict(meta)
    n, W, Hh = meta["n_frames"], meta["width"], meta["height"]
    net = np.full(n, np.nan)
    mv = cam_motion.sort_values("frame").reset_index(drop=True)
    fr_all = mv["frame"].to_numpy()
    is_new = mv["cut"].to_numpy() | (np.diff(fr_all, prepend=-10) != 1)
    shot_id = np.cumsum(is_new)

    p = df_det[df_det["type"] == "player"]
    p = p[(p["y2"] - p["y1"]) / Hh < closeup_h]
    px = ((p["x1"] + p["x2"]) / 2).to_numpy()
    rdf = {}
    for f, x in zip(p["frame"].to_numpy(), px):
        rdf.setdefault(int(f), []).append(x)
    # los 4 más confiables por frame para la regla 2|2
    p4 = p.sort_values(["frame", "conf"], ascending=[True, False]).groupby("frame").head(4)
    top4 = {}
    for f, x in zip(p4["frame"].to_numpy(), ((p4["x1"] + p4["x2"]) / 2).to_numpy()):
        top4.setdefault(int(f), []).append(x)

    info = []
    for pid in np.unique(shot_id):
        idx = np.where(shot_id == pid)[0]
        fr = fr_all[idx]
        dx = mv["dx"].to_numpy()[idx].copy()
        dx[0] = 0.0
        c = np.cumsum(dx)
        N, ref_source, n_ref = None, None, 0
        # 1) homografía
        if homographies:
            vals = [(net_x_in_image(homographies[f], Hh / 2) - cc)
                    for f, cc in zip(fr, c) if f in homographies]
            vals = np.array([v for v in vals if np.isfinite(v) and -W < v + 0 < 3 * W])
            if len(vals) >= min_frames_h:
                q1, q3 = np.percentile(vals, [25, 75])
                if q3 - q1 < 250:
                    N, ref_source, n_ref = float(np.median(vals)), "H", len(vals)
        # 2) jugadores
        if N is None:
            lo, hi, X = [], [], []
            for f, cc in zip(fr, c):
                xs4 = top4.get(int(f))
                if xs4 is not None and len(xs4) == 4:
                    xs = np.sort(xs4)
                    if np.argmax(np.diff(xs)) == 1:
                        lo.append(xs[1] - cc)
                        hi.append(xs[2] - cc)
                xs = rdf.get(int(f))
                if xs is not None:
                    X.extend(np.asarray(xs) - cc)
            if len(lo) >= min_intervals:
                lo, hi, X = np.array(lo), np.array(hi), np.array(X)
                grid = np.arange(lo.min(), hi.max() + 10, 10)
                cov = ((lo[:, None] <= grid) & (hi[:, None] >= grid)).sum(0)
                ok = cov >= 0.8 * cov.max()
                # zona contigua de máxima cobertura
                k = int(np.argmax(cov))
                a = k
                while a > 0 and ok[a - 1]:
                    a -= 1
                b = k
                while b < len(grid) - 1 and ok[b + 1]:
                    b += 1
                zone_lo, zone_hi = grid[a], grid[b]
                # hueco sin jugadores dentro de esa zona
                bins = np.arange(zone_lo, zone_hi + 25, 25)
                if len(bins) >= 2:
                    h, _ = np.histogram(X, bins=bins)
                    empty = h <= max(1, 0.01 * len(X))
                    best_run, start_t, run = (0, None), None, 0
                    for j, v in enumerate(empty):
                        if v:
                            start_t = j if run == 0 else start_t
                            run += 1
                            if run > best_run[0]:
                                best_run = (run, start_t)
                        else:
                            run = 0
                    if best_run[1] is not None:
                        N = float((bins[best_run[1]] + bins[best_run[1] + best_run[0]]) / 2)
                    else:
                        N = float((zone_lo + zone_hi) / 2)
                else:
                    N = float((zone_lo + zone_hi) / 2)
                ref_source, n_ref = "jugadores", len(lo)
            elif len(X) >= 30:
                # 3) pocos frames con los 4 jugadores (la cámara muestra media cancha):
                #    el hueco más ancho sin jugadores entre los dos grupos (~±200 px)
                X = np.array(X)
                p_lo, p_hi = np.percentile(X, [10, 90])
                bins = np.arange(p_lo, p_hi + 50, 50)
                if len(bins) >= 3:
                    h, _ = np.histogram(X, bins=bins)
                    empty = h <= max(1, 0.01 * len(X))
                    best_run, start_t, run = (0, None), None, 0
                    for j, v in enumerate(empty):
                        if v:
                            start_t = j if run == 0 else start_t
                            run += 1
                            if run > best_run[0]:
                                best_run = (run, start_t)
                        else:
                            run = 0
                    if best_run[1] is not None and best_run[0] >= 3:        # hueco >= 150 px
                        N = float((bins[best_run[1]] + bins[best_run[1] + best_run[0]]) / 2)
                        ref_source, n_ref = "hueco_jugadores", len(X)
        if N is not None:
            net[fr] = N + c
        info.append({"cam_shot": int(pid), "start": int(fr[0]), "end": int(fr[-1]),
                     "source": ref_source, "n_ref": n_ref, "N": None if N is None else round(N, 1)})
    info = pd.DataFrame(info)
    if verbose:
        totals = info.end - info.start + 1
        covered = totals[info.source.notna()].sum() / max(1, totals.sum())
        print(f"Red por paneo: {len(info)} planos, frames cubiertos {100 * covered:.0f}% "
              f"(H: {(info.source == 'H').sum()} planos, jugadores: {(info.source == 'jugadores').sum()})")
    return net, info


def net_from_players(df_det, meta, max_h=0.40, smooth_sec=0.5, fill_sec=1.0):
    """
    Posición x (imagen) de la red estimada SIN homografía: la cámara filma de costado,
    así que en la vista abierta los dos equipos quedan uno a cada lado de la red.
    Con los 4 jugadores más confiables (y no en primer plano), si se separan 2|2 en x,
    la red está en el promedio de las dos parejas. Validado contra la homografía:
    coincide en el lado de la pelota en ~95% de los frames (pelota a >40 px de la red).

    Args:
        df_det (pd.DataFrame): detecciones limpias (pelota + jugadores).
        meta (dict | pd.DataFrame): metadatos del video.
        max_h (float): altura máxima del jugador relativa al frame (más grande = primer plano).
        smooth_sec (float): ventana de suavizado (mediana móvil) en segundos.
        fill_sec (float): segundos máximos a rellenar hacia adelante/atrás.

    Returns:
        np.ndarray: largo n_frames con la x de la red (NaN donde no se puede estimar).
    """
    meta = meta_to_dict(meta)
    fps, n, H = meta["fps"], meta["n_frames"], meta["height"]
    p = df_det[df_det["type"] == "player"].copy()
    p["fx"] = (p["x1"] + p["x2"]) / 2
    p = p[(p["y2"] - p["y1"]) / H < max_h]
    p = p.sort_values(["frame", "conf"], ascending=[True, False]).groupby("frame").head(4)
    cnt = p.groupby("frame").size()
    p = p[p["frame"].isin(cnt[cnt == 4].index)].sort_values(["frame", "fx"])
    out = pd.Series(np.nan, index=range(n))
    if len(p):
        X = p["fx"].to_numpy().reshape(-1, 4)
        F = p["frame"].to_numpy().reshape(-1, 4)[:, 0]
        ok = np.argmax(np.diff(X, axis=1), axis=1) == 1        # separación 2 | 2
        out = pd.Series(np.where(ok, X.mean(1), np.nan), index=F).reindex(range(n))
    w = max(3, int(smooth_sec * fps))
    out = out.rolling(w, center=True, min_periods=5).median()
    lim = int(fill_sec * fps)
    return out.ffill(limit=lim).bfill(limit=lim).to_numpy()


def compensate_pan(traj, cam_motion, meta, smoothing=4):
    """
    Resta a la velocidad de la pelota la que agrega el paneo de la cámara (sin esto,
    un giro rápido de cámara parece un toque).

    Args:
        traj (pd.DataFrame): trayectoria de la pelota.
        cam_motion (pd.DataFrame): salida de camera_motion.
        meta (dict | pd.DataFrame): metadatos del video.
        smoothing (int): ventana (frames) de la media móvil del paneo.

    Returns:
        pd.DataFrame: copia de traj con vx y speed corregidos.
    """
    meta = meta_to_dict(meta)
    fps = meta["fps"]
    t = traj.copy()
    v = pd.Series(0.0, index=t.index)
    m = cam_motion.set_index("frame")["dx"]
    m = m[m.index.isin(t.index)]
    v.loc[m.index] = m.to_numpy()
    v = v.rolling(smoothing, center=True, min_periods=1).mean() * fps
    t["vx"] = t["vx"] - v
    t["speed"] = np.hypot(t["vx"], t["vy"])
    return t
