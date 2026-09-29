"""
Geometría de la cancha (1800x900, red en x=900): lados, línea de la red en la imagen,
y paso de imagen a cancha.
"""
import cv2
import numpy as np


SIDE_LEFT, SIDE_RIGHT = "izq", "der"          # mitad de cancha x<900 / x>=900
COURT_W, COURT_H, NET_X = 1800.0, 900.0, 900.0


def other_side(side):
    """
    Lado opuesto de la cancha.

    Args:
        side (str | None): SIDE_LEFT ('izq') o SIDE_RIGHT ('der').

    Returns:
        str | None: el otro lado, o None si side no es un lado válido.
    """
    return {SIDE_LEFT: SIDE_RIGHT, SIDE_RIGHT: SIDE_LEFT}.get(side)


def nearest_homography(homographies, sorted_frames, f, max_dist):
    """
    H del frame f o del más cercano a <= max_dist frames.

    Args:
        homographies (dict): {frame: H}.
        sorted_frames (np.ndarray): frames con homografía, ordenados.
        f (int): frame buscado.
        max_dist (int): distancia máxima en frames.

    Returns:
        np.ndarray | None: homografía 3x3, o None si no hay ninguna lo bastante cerca.
    """
    if f in homographies:
        return homographies[f]
    i = np.searchsorted(sorted_frames, f)
    best, bd = None, max_dist + 1
    for j in (i - 1, i):
        if 0 <= j < len(sorted_frames):
            d = abs(int(sorted_frames[j]) - f)
            if d < bd:
                best, bd = sorted_frames[j], d
    return homographies[best] if best is not None and bd <= max_dist else None


def _net_line_image(H):
    """
    Proyecta la red (x=900, de y=0 a y=900) a la imagen.

    Args:
        H (np.ndarray): homografía imagen -> cancha.

    Returns:
        tuple: (p_arriba, p_abajo, signo_izq): extremos de la red en la imagen y el signo
            del producto cruz que corresponde al lado izquierdo de la cancha.
    """
    Hinv = np.linalg.inv(H)
    pts = np.array([[[NET_X, 0.0]], [[NET_X, COURT_H]], [[NET_X / 2, COURT_H / 2]]], dtype=np.float32)
    img = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
    p0, p1, ref = img
    left_sign = np.sign(_cross(p0, p1, ref))
    return p0, p1, left_sign


def _cross(p0, p1, q):
    """
    Producto cruz 2D (p1 - p0) x (q - p0): de qué lado de la recta p0->p1 está q.

    Args:
        p0 (array-like): primer punto (x, y) de la recta.
        p1 (array-like): segundo punto (x, y) de la recta.
        q (array-like): punto a evaluar (x, y).

    Returns:
        float: valor con signo (0 = sobre la recta).
    """
    return (p1[0] - p0[0]) * (q[1] - p0[1]) - (p1[1] - p0[1]) * (q[0] - p0[0])


def distance_to_net(H, x, y):
    """
    Distancia con signo (px de imagen) de (x, y) a la línea de la red proyectada,
    positiva hacia el lado izquierdo de la cancha. Se mide en la IMAGEN contra la
    línea vertical de la red extendida: así una pelota en el aire se ubica bien
    aunque la homografía (que es del piso) no sirva para puntos en altura.

    Args:
        H (np.ndarray): homografía imagen -> cancha.
        x (float): coordenada x del punto en la imagen.
        y (float): coordenada y del punto en la imagen.

    Returns:
        float: distancia con signo en píxeles.
    """
    p0, p1, left_sign = _net_line_image(H)
    length = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) + 1e-9
    return left_sign * _cross(p0, p1, (x, y)) / length


def net_x_in_image(H, y):
    """
    x de la línea de la red (proyectada con H) a la altura y de la imagen.

    Args:
        H (np.ndarray): homografía imagen -> cancha.
        y (float): altura (px) en la imagen.

    Returns:
        float: x de la red en píxeles (NaN si la red proyectada es horizontal).
    """
    p0, p1, _ = _net_line_image(H)
    if abs(p1[1] - p0[1]) < 1e-6:
        return np.nan
    t = (y - p0[1]) / (p1[1] - p0[1])
    return float(p0[0] + t * (p1[0] - p0[0]))


def to_court(H, x, y):
    """
    Punto de imagen -> coordenadas de cancha (solo válido para puntos en el PISO).

    Args:
        H (np.ndarray): homografía imagen -> cancha.
        x (float): coordenada x en la imagen.
        y (float): coordenada y en la imagen.

    Returns:
        tuple: (x_cancha, y_cancha) en unidades de cancha (1800x900).
    """
    p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)[0][0]
    return float(p[0]), float(p[1])
