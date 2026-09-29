"""
Elementos de la transmisión que no son juego.

Repeticiones: la transmisión abre la repetición con una animación de la pelota gigante
amarilla/negra que llena la pantalla.
    detect_replay_graphics            -> animaciones por color (recorre el video, sin modelo)
    replay_graphics_from_detections   -> animaciones a partir de las detecciones (sin video)
    replay_mask_from_graphics         -> máscara de frames a ignorar

Cambios de vista (opcional): recorta el final de la jugada cuando la transmisión pasa a
zoom / primer plano / repetición.
    compute_frame_features + compute_off_view + trim_rallies_by_view
"""
import cv2
import numpy as np
import pandas as pd

from .common import find_runs, meta_to_dict


def _yellow_fraction(img):
    """
    Fracción de píxeles amarillo intenso (la animación ~25-60%; juego normal < 5%).

    Args:
        img (np.ndarray): frame BGR.

    Returns:
        float: fracción entre 0 y 1.
    """
    small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    y = (hsv[..., 0] >= 18) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 140) & (hsv[..., 2] >= 120)
    return float(y.mean())


def detect_replay_graphics(video_path, step=2, thr=0.15, verbose=True):
    """
    Recorre el video (sin modelo) y busca cada aparición de la animación de repetición.

    Args:
        video_path (str): ruta del video.
        step (int): analiza 1 de cada `step` frames.
        thr (float): fracción de amarillo a partir de la cual es la animación.
        verbose (bool): imprime el progreso.

    Returns:
        pd.DataFrame: start_frame, end_frame de cada animación.
    """
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hits, f = [], 0
    while True:
        if f % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            if _yellow_fraction(frame) > thr:
                hits.append(f)
        else:
            if not cap.grab():
                break
        if verbose and f % 20000 == 0:
            print(f"Buscando repeticiones: frame {f}/{n}")
        f += 1
    cap.release()

    events = []
    for h in hits:
        if events and h - events[-1][1] <= 3 * step:
            events[-1][1] = h
        else:
            events.append([h, h])
    ev = pd.DataFrame(events, columns=["start_frame", "end_frame"])
    if verbose:
        print(f"Animaciones de repetición encontradas: {len(ev)}")
    return ev


def replay_graphics_from_detections(df_det, meta, h_frac=0.9, min_frames=20, merge_sec=0.5):
    """
    Sin recorrer el video: la animación de la pelota gigante aparece en las detecciones
    como un "jugador" que ocupa casi toda la altura de la imagen durante ~0.7-1 s.

    Args:
        df_det (pd.DataFrame): detecciones.
        meta (dict | pd.DataFrame): metadatos del video.
        h_frac (float): altura relativa mínima de la caja.
        min_frames (int): frames mínimos para aceptar un evento.
        merge_sec (float): se unen frames separados por menos de esto.

    Returns:
        pd.DataFrame: start_frame, end_frame de cada evento.
    """
    meta = meta_to_dict(meta)
    fps, H = meta["fps"], meta["height"]
    p = df_det[df_det["type"] == "player"]
    fr = np.sort(p.loc[(p["y2"] - p["y1"]) >= h_frac * H, "frame"].unique())
    ev = []
    for f in fr:
        if ev and f - ev[-1][1] <= merge_sec * fps:
            ev[-1][1] = f
            ev[-1][2] += 1
        else:
            ev.append([f, f, 1])
    ev = [e for e in ev if e[2] >= min_frames]
    return pd.DataFrame([e[:2] for e in ev], columns=["start_frame", "end_frame"])


def replay_mask_from_graphics(events, meta, pair_max_sec=20.0, default_sec=1.0,
                              merge_sec=1.0, anim_max_sec=1.5):
    """
    Convierte las animaciones de repetición en una máscara de frames a ignorar.
    - Fragmentos separados por menos de merge_sec se unen (una misma animación).
    - Eventos de hasta anim_max_sec son la animación: si hay otra dentro de
      pair_max_sec (la que cierra la repetición), todo lo que hay entre ambas es
      repetición; si está suelta, solo se marca la animación + default_sec.
    - Eventos más largos que anim_max_sec no son la animación (pausas, publicidad
      con mucho amarillo): se marcan completos como "no juego" y no se emparejan.
    Sirve tanto para detect_replay_graphics (color, sin modelo) como para
    replay_graphics_from_detections.

    Args:
        events (pd.DataFrame): start_frame, end_frame de cada animación.
        meta (dict | pd.DataFrame): metadatos del video.
        pair_max_sec (float): separación máxima entre la animación de apertura y la de cierre.
        default_sec (float): segundos marcados tras una animación suelta.
        merge_sec (float): se unen fragmentos separados por menos de esto.
        anim_max_sec (float): duración máxima de una animación.

    Returns:
        np.ndarray: booleano de largo n_frames (True = repetición / no juego).
    """
    meta = meta_to_dict(meta)
    fps, n = meta["fps"], meta["n_frames"]
    mask = np.zeros(n, dtype=bool)
    ev = events.sort_values("start_frame")[["start_frame", "end_frame"]].to_numpy().tolist()
    merged = []
    for a, b in ev:
        if merged and a - merged[-1][1] <= merge_sec * fps:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    anims, breaks = [], []
    for a, b in merged:
        (anims if (b - a) <= anim_max_sec * fps else breaks).append((int(a), int(b)))
    for a, b in breaks:
        mask[a:min(n, b + 1)] = True
    i, windows = 0, 0
    while i < len(anims):
        a, b = anims[i]
        if i + 1 < len(anims) and anims[i + 1][0] - b <= pair_max_sec * fps:
            end = anims[i + 1][1]
            i += 2
        else:
            end = min(n - 1, b + int(default_sec * fps))
            i += 1
        mask[a:end + 1] = True
        windows += 1
    print(f"Tramos de repetición: {windows} + {len(breaks)} pausas largas "
          f"({mask.sum() / fps:.0f} s en total)")
    return mask


def _frame_hist(img, size=(160, 90)):
    """
    Histograma H-S normalizado de una versión reducida del frame.

    Args:
        img (np.ndarray): frame BGR.
        size (tuple): tamaño (ancho, alto) al que se reduce.

    Returns:
        np.ndarray: histograma 30x32 normalizado (float32).
    """
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).astype(np.float32)


def compute_frame_features(video_path, verbose=True):
    """
    Recorre el video SIN el modelo (rápido) y calcula, por frame, la diferencia de
    histograma con el frame anterior. Sirve para detectar transiciones de repetición.

    Args:
        video_path (str): ruta del video.
        verbose (bool): imprime el progreso.

    Returns:
        pd.DataFrame: frame, hist_diff (distancia de Bhattacharyya con el frame anterior).
    """
    cap = cv2.VideoCapture(video_path)
    feats, prev, f = [], None, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h = _frame_hist(frame)
        feats.append((f, 0.0 if prev is None else cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)))
        prev = h
        if verbose and f % 10000 == 0:
            print(f"Features frame {f}")
        f += 1
    cap.release()
    return pd.DataFrame(feats, columns=["frame", "hist_diff"])


def compute_off_view(df_det, meta, frame_feats=None,
                     max_player_h=0.40, min_players=2, smooth_sec=0.3,
                     gradual_thr=1.2, gradual_sec=0.3, cut_thr=0.45):
    """
    off_view[f] = True si el frame NO es la vista abierta del partido:
      - primer plano / zoom: el jugador más grande ocupa más de max_player_h del alto, o
      - menos de min_players jugadores detectados (tribuna, gráficos, jugador solo), o
      - transición gradual (wipe / logo de repetición), si se pasa frame_feats.
    Los cortes duros entre cámaras abiertas NO cuentan (pasan durante la jugada).

    Args:
        df_det (pd.DataFrame): detecciones.
        meta (dict | pd.DataFrame): metadatos del video.
        frame_feats (pd.DataFrame | None): salida de compute_frame_features.
        max_player_h (float): altura relativa máxima del jugador en vista abierta.
        min_players (int): jugadores mínimos en vista abierta.
        smooth_sec (float): ventana de suavizado (s).
        gradual_thr (float): suma de diferencias de histograma que indica una transición.
        gradual_sec (float): ventana (s) de esa suma.
        cut_thr (float): diferencia a partir de la cual es un corte duro (se ignora).

    Returns:
        np.ndarray: booleano de largo n_frames.
    """
    meta = meta_to_dict(meta)
    fps, n, H = meta["fps"], meta["n_frames"], meta["height"]

    p = df_det[df_det["type"] == "player"]
    n_pl = np.zeros(n)
    h_max = np.zeros(n)
    if not p.empty:
        g = p.assign(h=(p["y2"] - p["y1"]) / H).groupby("frame")["h"]
        idx = g.size().index.to_numpy()
        ok = idx < n
        n_pl[idx[ok]] = g.size().to_numpy()[ok]
        h_max[idx[ok]] = g.max().to_numpy()[ok]

    sw = max(1, int(smooth_sec * fps))
    n_s = pd.Series(n_pl).rolling(sw, center=True, min_periods=1).max().to_numpy()
    h_s = pd.Series(h_max).rolling(sw, center=True, min_periods=1).median().to_numpy()
    off = (n_s < min_players) | (h_s > max_player_h)

    if frame_feats is not None:
        hd = np.zeros(n)
        ff = frame_feats[frame_feats["frame"] < n]
        hd[ff["frame"].to_numpy()] = ff["hist_diff"].to_numpy()
        hd_soft = np.where(hd > cut_thr, 0.0, hd)  # ignorar cortes duros
        win = max(2, int(gradual_sec * fps))
        acc = pd.Series(hd_soft).rolling(win, min_periods=1).sum().to_numpy()
        trans = acc > gradual_thr
        # Lo que viene justo después de una transición suele ser la repetición
        ext = int(1.0 * fps)
        trans = pd.Series(trans.astype(float)).rolling(ext, min_periods=1).max().to_numpy() > 0
        off |= trans

    print(f"Frames fuera de la vista de juego: {off.mean():.1%}")
    return off


def trim_rallies_by_view(rallies, off_view, meta, min_off_sec=0.5, keep_after_off_sec=0.0):
    """
    Recorta el FINAL de cada jugada: busca, después de que la jugada ya se ve en plano
    abierto, el primer tramo fuera de la vista de juego que dure al menos min_off_sec
    (zoom a jugadores, primer plano, repetición) y corta ahí.
    No modifica el inicio (el saque puede verse en primer plano) ni jugadas sin ese tramo.

    Args:
        rallies (pd.DataFrame): jugadas.
        off_view (np.ndarray): salida de compute_off_view.
        meta (dict | pd.DataFrame): metadatos del video.
        min_off_sec (float): duración mínima del tramo fuera de vista.
        keep_after_off_sec (float): segundos que se conservan después del inicio del tramo.

    Returns:
        pd.DataFrame: copia de rallies con end_frame / clip_end recortados y columna trimmed.
    """
    meta = meta_to_dict(meta)
    if rallies.empty:
        return rallies
    fps = meta["fps"]
    n = len(off_view)
    min_off = max(1, int(min_off_sec * fps))
    keep = int(keep_after_off_sec * fps)

    df = rallies.copy()
    df["trimmed"] = False
    for i, r in df.iterrows():
        s0, ce = int(r["serve_frame"]), int(min(r["clip_end"], n - 1))
        # Primer frame en vista abierta desde el saque
        on = np.where(~off_view[s0:ce + 1])[0]
        if len(on) == 0:
            continue
        f0 = s0 + on[0]
        for a, b in find_runs(off_view[f0:ce + 1]):
            # b == fin del recorte: el tramo puede seguir más allá de clip_end
            full_b = b
            while f0 + full_b < n and off_view[f0 + full_b]:
                full_b += 1
            if full_b - a >= min_off:
                cut = f0 + a + keep
                if cut <= r["clip_end"]:
                    df.at[i, "clip_end"] = cut - 1
                    df.at[i, "end_frame"] = min(int(r["end_frame"]), cut - 1)
                    df.at[i, "trimmed"] = True
                break

    df["duration_sec"] = ((df["end_frame"] - df["serve_frame"]) / fps).round(2)
    print(f"Jugadas recortadas por cambio de vista: {int(df['trimmed'].sum())} de {len(df)}")
    return df
