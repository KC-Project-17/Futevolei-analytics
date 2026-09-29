"""
Trayectoria de la pelota: UNA pelota por frame, huecos cortos interpolados, suavizado y
velocidades (px/s normalizados a 1080p; vy > 0 = cayendo).
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .common import find_runs, meta_to_dict


def _interp_short_gaps(x, max_gap):
    """
    Interpola linealmente solo los huecos de hasta max_gap frames.

    Args:
        x (np.ndarray): serie con NaN en los huecos.
        max_gap (int): largo máximo de hueco a interpolar.

    Returns:
        np.ndarray: serie con los huecos cortos rellenados.
    """
    s = pd.Series(x)
    filled = s.interpolate(limit_area="inside")
    isn = s.isna()
    run_len = isn.groupby((~isn).cumsum()).transform("sum")
    filled[isn & (run_len > max_gap)] = np.nan
    return filled.to_numpy()


def _smooth_runs(x, window, deriv=0):
    """
    Savitzky-Golay por tramos continuos (no cruza huecos).

    Args:
        x (np.ndarray): serie con NaN en los huecos.
        window (int): ventana del filtro (impar).
        deriv (int): 0 = suavizado, 1 = derivada (por frame).

    Returns:
        np.ndarray: serie suavizada (o su derivada), NaN en los huecos.
    """
    out = np.full_like(x, np.nan, dtype=float)
    for a, b in find_runs(~np.isnan(x)):
        seg = x[a:b]
        n = b - a
        w = min(window, n if n % 2 == 1 else n - 1)
        if w >= 5:
            out[a:b] = savgol_filter(seg, w, 2, deriv=deriv)
        elif deriv == 0:
            out[a:b] = seg
        else:
            out[a:b] = np.gradient(seg) if n > 1 else 0.0
    return out


def build_ball_trajectory(df_det, meta, max_speed_px_s=2500, reacquire_sec=0.5,
                          max_interp_sec=0.25, smooth_window=9):
    """
    Construye una única trayectoria de pelota (una posición por frame):
    - Asociación por vecino más cercano a la última posición, con compuerta de velocidad.
    - Si se pierde más de reacquire_sec, se re-adquiere con la detección de mayor confianza.
    - Interpola huecos cortos, suaviza y calcula velocidades (px/s normalizados a 1080p).

    Args:
        df_det (pd.DataFrame): detecciones limpias.
        meta (dict | pd.DataFrame): metadatos del video.
        max_speed_px_s (float): velocidad máxima creíble (px/s a 1080p).
        reacquire_sec (float): tiempo sin pelota tras el cual se re-adquiere.
        max_interp_sec (float): hueco máximo (s) que se interpola.
        smooth_window (int): ventana del suavizado Savitzky-Golay.

    Returns:
        pd.DataFrame: frame, detected, reacq_jump, cx, cy, vx, vy, speed (vy > 0 = cayendo).
    """
    meta = meta_to_dict(meta)
    fps, n = meta["fps"], meta["n_frames"]
    scale = meta["height"] / 1080
    max_step = max_speed_px_s * scale / fps
    reacq = int(reacquire_sec * fps)

    balls = df_det[df_det["type"] == "ball"]
    by_frame = {f: g[["cx", "cy", "conf"]].to_numpy() for f, g in balls.groupby("frame")}

    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    jump = np.full(n, np.nan)  # en re-adquisiciones: distancia (px 1080p) a la última posición conocida
    last, last_f = None, None
    for f in sorted(by_frame):
        cand = by_frame[f]
        if last is not None and f - last_f <= reacq:
            d = np.hypot(cand[:, 0] - last[0], cand[:, 1] - last[1])
            j = int(np.argmin(d))
            if d[j] > max_step * (f - last_f):
                continue  # salto imposible: probablemente falso positivo
        else:
            j = int(np.argmax(cand[:, 2]))
            jump[f] = np.inf if last is None else np.hypot(cand[j, 0] - last[0], cand[j, 1] - last[1]) / scale
        cx[f], cy[f] = cand[j, 0], cand[j, 1]
        last, last_f = (cx[f], cy[f]), f

    detected = ~np.isnan(cx)
    max_gap = int(max_interp_sec * fps)
    cx_i = _interp_short_gaps(cx, max_gap)
    cy_i = _interp_short_gaps(cy, max_gap)

    traj = pd.DataFrame({
        "frame": np.arange(n),
        "detected": detected,
        "reacq_jump": jump,
        "cx": _smooth_runs(cx_i, smooth_window),
        "cy": _smooth_runs(cy_i, smooth_window),
        # Derivadas en px/frame -> px/s normalizados a 1080p. vy > 0 = cayendo.
        "vx": _smooth_runs(cx_i, smooth_window, deriv=1) * fps / scale,
        "vy": _smooth_runs(cy_i, smooth_window, deriv=1) * fps / scale,
    })
    traj["speed"] = np.hypot(traj["vx"], traj["vy"])
    print(f"Pelota detectada en {detected.mean():.1%} de los frames "
          f"({(~traj['cy'].isna()).mean():.1%} tras interpolar)")
    return traj
