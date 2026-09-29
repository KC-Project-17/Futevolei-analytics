"""
Eventos de la pelota dentro de una jugada: toques (cambios bruscos de velocidad que la
gravedad no explica), lado de la pelota en cada frame y cruces de la red.
"""
import numpy as np

from .common import meta_to_dict
from .court_geometry import SIDE_LEFT, SIDE_RIGHT, distance_to_net, nearest_homography


def detect_touches(traj, start, end, meta, dv_min=650, win_frames=5, sep_sec=0.25,
                    g_px=2000.0, min_det=2, v_min_after=300):
    """
    Frames de contacto dentro de [start, end]. Un toque cambia la velocidad de la
    pelota de golpe (hacia arriba, de lado o hacia abajo en un remate). Se compara
    la velocidad media antes y después de cada frame y se descuenta lo que explica
    la gravedad. Velocidades en px/s normalizados a 1080p (como traj).

    Args:
        traj (pd.DataFrame): trayectoria de la pelota (build_ball_trajectory), con vx, vy, detected.
        start (int): primer frame de la jugada.
        end (int): último frame de la jugada.
        meta (dict | pd.DataFrame): metadatos del video.
        dv_min (float): cambio de velocidad mínimo (px/s) para contar un toque.
        win_frames (int): frames antes/después usados para la velocidad media.
        sep_sec (float): separación mínima entre toques (s).
        g_px (float): gravedad en px/s^2 (a 1080p).
        min_det (int): detecciones mínimas en cada ventana.
        v_min_after (float): velocidad mínima después del contacto; si la pelota queda
            casi quieta, fue un golpe contra la arena y no un toque.

    Returns:
        list: frames de los toques, ordenados.
    """
    meta = meta_to_dict(meta)
    fps = meta["fps"]
    vx = traj["vx"].to_numpy()
    vy = traj["vy"].to_numpy()
    det = traj["detected"].to_numpy()
    w = win_frames
    dt = (w + 1) / fps
    cand = []
    for t in range(start + w + 1, end - w):
        a = slice(t - w - 1, t - 1)
        b = slice(t + 1, t + w + 1)
        if det[a].sum() < min_det or det[b].sum() < min_det:
            continue
        va = np.array([np.nanmean(vx[a]), np.nanmean(vy[a])])
        vb = np.array([np.nanmean(vx[b]), np.nanmean(vy[b])])
        if np.isnan(va).any() or np.isnan(vb).any():
            continue
        dv = vb - va - np.array([0.0, g_px * dt])
        cand.append((t, float(np.hypot(*dv))))
    # máximos locales separados al menos sep_sec
    sp = np.hypot(vx, vy)
    touches = []
    sep = int(sep_sec * fps)
    for t, m in sorted(cand, key=lambda x: -x[1]):
        if m < dv_min:
            break
        if not all(abs(t - u) >= sep for u in touches):
            continue
        # Golpe contra la arena: la pelota queda casi quieta después -> no es un toque.
        speed_after = sp[t + w:t + w + int(0.3 * fps)]        # 0.3 s después del contacto
        if (~np.isnan(speed_after)).any() and np.nanmedian(speed_after) < v_min_after:
            continue
        touches.append(t)
    return sorted(touches)


def side_per_frame(traj, start, end, homographies, frames_h, meta, margin_px=15, max_dist_h=15,
                   net_players_x=None, player_margin_px=40, net_pan_x=None, pan_margin_px=80):
    """
    Lado de la pelota en cada frame de la jugada. Mantiene el último lado si la
    pelota está cerca de la red (evita cruces falsos por ruido).
    Prioridad: red por paneo (si se pasó) -> homografía -> red por jugadores.
    Con red por paneo solo se usan frames con la pelota DETECTADA (las posiciones
    interpoladas no sirven cuando la cámara gira).

    Args:
        traj (pd.DataFrame): trayectoria de la pelota (cx, cy, detected).
        start (int): primer frame.
        end (int): último frame.
        homographies (dict): {frame: H} imagen -> cancha.
        frames_h (np.ndarray): frames con homografía, ordenados.
        meta (dict | pd.DataFrame): metadatos del video.
        margin_px (float): distancia mínima a la red (px a 1080p) con homografía.
        max_dist_h (int): distancia máxima (frames) para usar la H más cercana.
        net_players_x (np.ndarray | None): x de la red por frame según net_from_players.
        player_margin_px (float): margen con la red estimada por jugadores.
        net_pan_x (np.ndarray | None): x de la red por frame según net_from_pan.
        pan_margin_px (float | np.ndarray): margen con la red por paneo; número o array por
            frame (más margen donde la red se estimó con los jugadores, que es menos precisa).

    Returns:
        list: largo end-start+1 con 'izq' / 'der' / None. Además deja en
            side_per_frame.ref_source la referencia usada en cada frame ('paneo', 'H',
            'jugadores' o None).
    """
    meta = meta_to_dict(meta)
    s = meta["height"] / 1080
    cx = traj["cx"].to_numpy()
    cy = traj["cy"].to_numpy()
    det = traj["detected"].to_numpy().astype(bool)
    sides = [None] * (end - start + 1)
    ref_source = [None] * (end - start + 1)
    current = None
    for k, f in enumerate(range(start, end + 1)):
        if net_pan_x is not None and not np.isnan(net_pan_x[f]):
            if det[f] and not np.isnan(cx[f]):
                d = net_pan_x[f] - cx[f]
                ref_source[k] = "paneo"
                margin_px_f = pan_margin_px[f] if np.ndim(pan_margin_px) else pan_margin_px
                if abs(d) > margin_px_f * s:
                    current = SIDE_LEFT if d > 0 else SIDE_RIGHT
        elif not np.isnan(cx[f]) and (net_pan_x is None or det[f]):
            H = nearest_homography(homographies, frames_h, f, max_dist_h)
            if H is not None:
                d = distance_to_net(H, cx[f], cy[f])
                ref_source[k] = "H"
                if abs(d) > margin_px * s:
                    current = SIDE_LEFT if d > 0 else SIDE_RIGHT
            elif net_players_x is not None and not np.isnan(net_players_x[f]):
                d = net_players_x[f] - cx[f]
                ref_source[k] = "jugadores"
                if abs(d) > player_margin_px * s:
                    current = SIDE_LEFT if d > 0 else SIDE_RIGHT
        sides[k] = current
    side_per_frame.ref_source = ref_source
    return sides


def clean_sides(sides, s0, touches, min_frames):
    """
    Une al tramo anterior los tramos de lado intermedios (ni el primero ni el último)
    que no contienen ningún toque y duran menos de min_frames. La pelota no puede
    cruzar la red y volver sin que nadie la toque: eso es ruido de la línea de la red.

    Args:
        sides (list): lado por frame ('izq' / 'der' / None), desde s0.
        s0 (int): frame correspondiente a sides[0].
        touches (list): frames de los toques.
        min_frames (int): largo mínimo de un tramo sin toques para conservarlo.

    Returns:
        list: lados corregidos.
    """
    sides = list(sides)
    tset = set(t - s0 for t in touches)
    changed = True
    while changed:
        changed = False
        # tramos de valor constante (ignorando None al inicio)
        segments = []
        k = 0
        while k < len(sides):
            if sides[k] is None:
                k += 1
                continue
            j = k
            while j + 1 < len(sides) and (sides[j + 1] == sides[k] or sides[j + 1] is None):
                j += 1
            segments.append((k, j, sides[k]))
            k = j + 1
        for i in range(1, len(segments) - 1):
            a, b, l = segments[i]
            no_touch = not any(a <= t <= b for t in tset)
            if no_touch and (b - a + 1) < min_frames and segments[i - 1][2] == segments[i + 1][2]:
                for q in range(a, b + 1):
                    sides[q] = segments[i - 1][2]
                changed = True
                break
    return sides


def crossings_from_sides(sides, s0):
    """
    Cruces de red a partir del lado por frame.

    Args:
        sides (list): lado por frame ('izq' / 'der' / None).
        s0 (int): frame correspondiente a sides[0].

    Returns:
        list: tuplas (frame, desde, hacia) de cada cruce.
    """
    crossings, prev = [], None
    for k, l in enumerate(sides):
        if l is not None and prev is not None and l != prev:
            crossings.append((s0 + k, prev, l))     # (frame, desde, hacia)
        if l is not None:
            prev = l
    return crossings
