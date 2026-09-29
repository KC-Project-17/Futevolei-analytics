"""
Homografía imagen -> cancha por frame a partir del modelo de keypoints de la cancha
(Caso A: 4+ keypoints; Caso B: 3 keypoints + movimiento rígido; pooling por tipo de
plano y bootstrap hacia atrás).
"""
import cv2
import numpy as np
from collections import defaultdict

# =====================================================================
# Preprocesamiento consistente para el modelo de keypoints de cancha.
#
# El modelo fue entrenado con frames redimensionados a 1280x1280 SIN
# preservar aspect ratio (cv2.resize "achatando" la imagen a cuadrado).
# El resize interno de Ultralytics (rect=True o rect=False) hace
# letterbox -preserva aspect ratio, rellena con padding- que es una
# distribución de píxeles DISTINTA a la del entrenamiento. Si se le da
# el frame original directo a predict(), el modelo recibe una imagen
# preprocesada de forma diferente a como aprendió, y los keypoints
# salen sistemáticamente mal ubicados -no es un bug de coordenadas, es
# un mismatch de preprocesamiento train/inferencia.
#
# Por eso: SIEMPRE reproducir el resize manual a cuadrado (matchea
# entrenamiento) para la inferencia, y reescalar el resultado a
# resolución original con factores NO uniformes (scale_x, scale_y por
# separado, porque el resize no preserva aspect ratio) antes de usar
# esos keypoints para cualquier otra cosa (homografía, dibujo, etc.).
# Usar esta misma función en TODO el pipeline (visualización,
# rally_homographies.py) para que no
# vuelva a haber inconsistencia entre scripts.
# =====================================================================


def predict_court_keypoints(model, frame, input_size=1280, conf=0.1, iou=0.5):
    """
    Corre el modelo de pose sobre 'frame' replicando el preprocesamiento
    de entrenamiento (resize a input_size x input_size sin preservar aspect
    ratio), y devuelve los keypoints/confianzas de la detección de mayor
    confianza YA reescalados a la resolución original de 'frame'.

    Args:
        model: modelo YOLO de pose (keypoints de cancha).
        frame (np.ndarray): imagen BGR en resolución original.
        input_size (int): lado del cuadrado al que se redimensiona (el de entrenamiento).
        conf (float): confianza mínima de detección para predict().
        iou (float): umbral de IoU del NMS.

    Returns:
        tuple: (kpts, confs) o (None, None) si no se detectó nada.
            kpts: array (6, 2) en píxeles de la resolución original de 'frame'.
            confs: array (6,) con la confianza de cada keypoint.
    """
    h_orig, w_orig = frame.shape[:2]

    frame_resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Factores de rescale NO uniformes -el resize achata, no preserva
    # aspect ratio, así que scale_x y scale_y son distintos en general.
    scale_x = w_orig / input_size
    scale_y = h_orig / input_size

    results = model.predict(
        source=frame_rgb, imgsz=input_size, conf=conf, iou=iou,
        rect=False, verbose=False, show_boxes=False,
    )

    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None, None

    result = results[0]
    best_det_idx = 0
    if result.boxes is not None and len(result.boxes.conf) > 0:
        best_det_idx = int(result.boxes.conf.argmax())

    kp_xy = result.keypoints.xy[best_det_idx].cpu().numpy()
    kp_conf = result.keypoints.conf[best_det_idx].cpu().numpy()

    kpts_orig = kp_xy * np.array([scale_x, scale_y], dtype=np.float32)
    return kpts_orig, kp_conf




KEYPOINT_NAMES = ["esquina_sup_izq", "red_sup", "esquina_sup_der", "esquina_inf_der", "red_inf", "esquina_inf_izq"]

# =====================================================================
# DEFINICIÓN DE LA CANCHA REAL Y ESTIMACIÓN ROBUSTA DE PERSPECTIVA
# =====================================================================

# Proporciones estándar oficiales de la cancha.
# IMPORTANTE: el orden de KEYPOINT_NAMES tiene que coincidir
# EXACTAMENTE con el orden en que el modelo de pose fue entrenado
# (kpts[0] debe ser esquina_sup_izq, kpts[1] red_sup, etc.). Si el
# dataset se retocó (por ejemplo al migrar a COCO Keypoints), verificá
# una vez con un frame de ejemplo que la asignación siga siendo
# correcta -un desorden acá no tira error, simplemente mapea cada punto
# al lugar equivocado de la cancha de forma consistente y silenciosa.
REAL_COURT_MAP = {
    "esquina_sup_izq": np.array([0, 0], dtype=np.float32),
    "esquina_sup_der": np.array([1800, 0], dtype=np.float32),
    "red_sup":         np.array([900, 0], dtype=np.float32),
    "red_inf":         np.array([900, 900], dtype=np.float32),
    "esquina_inf_izq": np.array([0, 900], dtype=np.float32),
    "esquina_inf_der": np.array([1800, 900], dtype=np.float32),
}

# Las dos mitades de la cancha (orden TL, TR, BR, BL en cada una, igual
# que el orden usado para la cancha completa). Cuando solo hay 3
# keypoints visibles, sirven para saber a qué mitad pertenecen y
# estimar el 4to punto DENTRO de esa mitad -nunca uno del lado que la
# cámara no está mostrando.
LEFT_HALF = ["esquina_sup_izq", "red_sup", "red_inf", "esquina_inf_izq"]
RIGHT_HALF = ["red_sup", "esquina_sup_der", "esquina_inf_der", "red_inf"]
# Las 4 esquinas REALES de la cancha completa (sin puntos de red) -para
# el caso de un plano de cancha completa (no zoomeado a una mitad)
# donde por oclusión falta una esquina, pero las otras 3 sí están.
FULL_CORNERS = ["esquina_sup_izq", "esquina_sup_der", "esquina_inf_der", "esquina_inf_izq"]

# Cuántos frames de distancia desde la última actualización real se
# tolera al arrastrar una semilla en el bootstrap hacia atrás. Más allá
# de esto, mejor dejar el frame sin homografía que una desactualizada.
MAX_BOOTSTRAP_DRAG_FRAMES = 30

# Para el pooling por tipo de plano: mínimo de frames para confiar en
# un tipo de plano como "fijo", y dispersión máxima aceptable (px) al
# medir cuánto varía la proyección del centro de la cancha entre las
# homografías individuales de esos frames.
MIN_FRAMES_POOL = 30
POOL_SPREAD_THRESHOLD_PX = 15.0


def validate_geometry(H, canonical_corners, img_w, img_h):
    """
    Valida que la homografía genere una forma con sentido geométrico,
    para el conjunto de esquinas canónicas que se le pase (cancha
    entera o una mitad).

    Args:
        H (np.ndarray | None): homografía 3x3 imagen -> cancha.
        canonical_corners (list): esquinas en coordenadas de cancha, en orden TL, TR, BR, BL.
        img_w (int): ancho del frame en píxeles.
        img_h (int): alto del frame en píxeles.

    Returns:
        bool: True si la proyección es convexa, con área razonable y bien orientada.
    """
    if H is None:
        return False

    corners = np.array(canonical_corners, dtype=np.float32).reshape(-1, 1, 2)
    try:
        H_inv = np.linalg.inv(H)
        pts = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
    except Exception:
        return False

    if not cv2.isContourConvex(np.int32(pts)):
        return False

    area = cv2.contourArea(np.int32(pts))
    area_total = img_w * img_h
    if area < area_total * 0.05 or area > area_total * 2.0:
        return False

    if pts[0][1] >= pts[3][1] or pts[1][1] >= pts[2][1]:
        return False

    return True


def validate_court_geometry(H, img_w, img_h):
    """
    Valida contra la cancha ENTERA (Caso A, 4+ puntos).

    Args:
        H (np.ndarray | None): homografía 3x3 imagen -> cancha.
        img_w (int): ancho del frame en píxeles.
        img_h (int): alto del frame en píxeles.

    Returns:
        bool: True si la geometría es válida.
    """
    return validate_geometry(H, [[0, 0], [1800, 0], [1800, 900], [0, 900]], img_w, img_h)


def validate_quadrilateral_geometry(H, quad, img_w, img_h):
    """
    Valida contra el rectángulo de UN CUADRILÁTERO dado (una mitad, o
    las 4 esquinas reales de la cancha completa) -Caso B, 3 puntos.

    Args:
        H (np.ndarray | None): homografía 3x3 imagen -> cancha.
        quad (list): nombres de los 4 keypoints del cuadrilátero (LEFT_HALF, RIGHT_HALF o FULL_CORNERS).
        img_w (int): ancho del frame en píxeles.
        img_h (int): alto del frame en píxeles.

    Returns:
        bool: True si la geometría es válida.
    """
    corners = [REAL_COURT_MAP[n].tolist() for n in quad]
    return validate_geometry(H, corners, img_w, img_h)


def detect_quadrilateral(visible_points):
    """
    A qué cuadrilátero de 4 puntos pertenecen los 3 visibles, si
    encajan limpiamente en uno. Prueba, en orden: mitad izquierda,
    mitad derecha, y las 4 esquinas reales de la cancha completa (para
    el caso de un plano de cancha completa donde falta una esquina por
    oclusión, no un punto de red). None si no encajan en ninguno -por
    ejemplo, 2 esquinas de lados opuestos más 1 punto de red, una
    combinación ambigua que no corresponde a ningún cuadrilátero
    conocido con un solo punto faltante.

    Args:
        visible_points (dict): {nombre_keypoint: (x, y)} de los puntos visibles.

    Returns:
        list | None: LEFT_HALF, RIGHT_HALF, FULL_CORNERS o None.
    """
    visible_names = set(visible_points.keys())
    if visible_names <= set(LEFT_HALF):
        return LEFT_HALF
    if visible_names <= set(RIGHT_HALF):
        return RIGHT_HALF
    if visible_names <= set(FULL_CORNERS):
        return FULL_CORNERS
    return None


# =====================================================================
# HOMOGRAFÍAS POOLED POR TIPO DE PLANO
#
# Si la cámara vuelve repetidamente al mismo encuadre a lo largo del
# video (mitad izquierda / centro / mitad derecha), cada vez que ese
# encuadre aparece es geométricamente la misma vista. En vez de confiar
# en el frame temporalmente más cercano (que puede estar arrastrando
# drift), buscamos entre TODOS los tipos de plano ya identificados como
# fijos -sin importar cuán lejos en el tiempo estén- cuál es compatible
# con los puntos que se ven ahora, y usamos esa homografía (ajustada
# con las correspondencias de MUCHOS frames) como referencia.
# =====================================================================

def visible_keypoints(confs, conf_thresh):
    """
    Nombres de los keypoints con confianza suficiente.

    Args:
        confs (array): confianza de cada keypoint (orden KEYPOINT_NAMES).
        conf_thresh (float): confianza mínima para considerarlo visible.

    Returns:
        frozenset: nombres de los keypoints visibles (sirve de "firma" del plano).
    """
    return frozenset(n for n, c in zip(KEYPOINT_NAMES, confs) if c >= conf_thresh)


def compute_pooled_homographies(kpts_per_frame, img_w, img_h, conf_thresh):
    """
    Agrupa frames por qué keypoints ven con confianza. Para los grupos
    con suficientes frames y baja dispersión (plano realmente fijo, no
    solo parecido), ajusta UNA homografía robusta pooleando las
    correspondencias de todos esos frames.

    Args:
        kpts_per_frame (dict): {frame: (kpts, confs)} con los keypoints de cada frame.
        img_w (int): ancho del frame en píxeles.
        img_h (int): alto del frame en píxeles.
        conf_thresh (float): confianza mínima para considerar un keypoint visible.

    Returns:
        dict: {firma (frozenset de nombres): H (np.ndarray 3x3)}.
    """
    groups = defaultdict(list)
    for f, (kpts, confs) in kpts_per_frame.items():
        signature = visible_keypoints(confs, conf_thresh)
        if len(signature) >= 4:
            groups[signature].append(f)

    canonical_center = np.array([[[900, 450]]], dtype=np.float32)
    pooled = {}

    for signature, group_frames in groups.items():
        if len(group_frames) < MIN_FRAMES_POOL:
            continue

        projections = []
        for f in group_frames:
            kpts, confs = kpts_per_frame[f]
            src = np.array([kpts[KEYPOINT_NAMES.index(n)] for n in signature], dtype=np.float32)
            dst = np.array([REAL_COURT_MAP[n] for n in signature], dtype=np.float32)
            H_frame, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H_frame is None:
                continue
            try:
                p = cv2.perspectiveTransform(canonical_center, np.linalg.inv(H_frame))[0][0]
                projections.append(p)
            except np.linalg.LinAlgError:
                continue

        if len(projections) < 5:
            continue
        dispersion = float(np.mean(np.std(np.array(projections), axis=0)))
        if dispersion > POOL_SPREAD_THRESHOLD_PX:
            continue

        src_pooled, dst_pooled = [], []
        for f in group_frames:
            kpts, confs = kpts_per_frame[f]
            for n in signature:
                src_pooled.append(kpts[KEYPOINT_NAMES.index(n)])
                dst_pooled.append(REAL_COURT_MAP[n])
        src_pooled = np.array(src_pooled, dtype=np.float32)
        dst_pooled = np.array(dst_pooled, dtype=np.float32)
        H_pool, _ = cv2.findHomography(src_pooled, dst_pooled, cv2.RANSAC, 3.0)
        if H_pool is not None:
            pooled[signature] = H_pool

    return pooled


def find_compatible_pooled_homography(current_visible_names, pooled_homographies):
    """
    De las homografías pooled, la que contenga (como superset) los
    puntos actualmente visibles. Si hay varias, la de firma más chica
    (la correspondencia más cercana al plano actual -menos puntos
    'extra' que no se están viendo ahora).

    Args:
        current_visible_names (frozenset): keypoints visibles en el frame actual.
        pooled_homographies (dict): salida de compute_pooled_homographies.

    Returns:
        np.ndarray | None: homografía compatible o None si no hay ninguna.
    """
    candidates = [
        (signature, H) for signature, H in pooled_homographies.items() if current_visible_names <= signature
    ]
    if not candidates:
        return None
    _, H = min(candidates, key=lambda x: len(x[0]))
    return H


def estimate_point_with_motion(id_missing, visible_points, H_last):
    """
    Estima la ubicación actual de un keypoint faltante calculando una
    transformación rígida parcial (traslación, rotación, escala) sobre
    los 3 puntos visibles.

    LIMITACIÓN A TENER EN CUENTA: esto asume movimiento de cámara
    RÍGIDO entre el frame de H_last y el actual. Si H_last viene de
    muchos frames atrás (racha larga sin homografía directa), el
    supuesto de movimiento rígido deja de sostenerse -la cámara puede
    haber hecho un cambio de perspectiva real, no solo pan/rotación- y
    el punto estimado se aleja cada vez más de la posición real cuanto
    más vieja es esa H_last. Si medís rachas largas, conviene desconfiar
    de las homografías del Caso B que caen dentro de una racha extensa.

    Args:
        id_missing (str): nombre del keypoint faltante.
        visible_points (dict): {nombre: (x, y)} con exactamente 3 puntos visibles.
        H_last (np.ndarray | None): última homografía válida (imagen -> cancha).

    Returns:
        np.ndarray | None: posición (x, y) estimada en píxeles, o None si no se puede estimar.
    """
    if H_last is None or len(visible_points) != 3:
        return None

    try:
        H_inv = np.linalg.inv(H_last)
    except np.linalg.LinAlgError:
        return None

    pts_old = []
    pts_new = []
    for id_vis, pt_new in visible_points.items():
        p_map = np.array([[REAL_COURT_MAP[id_vis]]], dtype=np.float32)
        p_old = cv2.perspectiveTransform(p_map, H_inv)[0][0]
        pts_old.append(p_old)
        pts_new.append(pt_new)

    pts_old = np.array(pts_old, dtype=np.float32)
    pts_new = np.array(pts_new, dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(pts_old, pts_new)

    if M is None:
        dx = np.mean(pts_new[:, 0] - pts_old[:, 0])
        dy = np.mean(pts_new[:, 1] - pts_old[:, 1])
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

    p_map_missing = np.array([[REAL_COURT_MAP[id_missing]]], dtype=np.float32)
    p_missing_old = cv2.perspectiveTransform(p_map_missing, H_inv)[0][0]
    p_missing_new = np.dot(M, np.array([p_missing_old[0], p_missing_old[1], 1.0]))

    return p_missing_new


def _estimate_is_plausible(half_quad, visible_points, id_missing, p_est,
                              cos_tolerance=0.0, ratio_min=0.2, ratio_max=5.0):
    """
    Chequeo INDEPENDIENTE del punto estimado (no depende de reproyectar
    con la homografía recién ajustada -eso sería circular). Compara los
    dos lados que tocan al punto faltante contra sus lados OPUESTOS, que
    están completamente formados por los 3 puntos REALES (siempre alcanza,
    porque solo falta 1 de los 4). En un rectángulo recorrido en orden
    cíclico (TL->TR->BR->BL->TL), cada lado va en sentido CONTRARIO a su
    lado opuesto -si el lado que toca al punto estimado sale en el MISMO
    sentido que su opuesto en vez de contrario, el punto quedó "para atrás"
    en vez de hacia donde correspondía.

    Umbrales relajados por defecto (cos_tolerance=0.0 -alcanza con que sea
    negativo, no exige que sea MARCADAMENTE negativo; ratio 0.2-5.0 -más
    permisivo con perspectiva pronunciada).

    Args:
        half_quad (list): nombres de los 4 keypoints del cuadrilátero, en orden cíclico.
        visible_points (dict): {nombre: (x, y)} de los 3 puntos visibles.
        id_missing (str): nombre del keypoint estimado.
        p_est (np.ndarray): posición estimada (x, y) del keypoint faltante.
        cos_tolerance (float): cuánto de negativo tiene que ser el coseno entre lados opuestos.
        ratio_min (float): relación mínima de longitudes entre lados opuestos.
        ratio_max (float): relación máxima de longitudes entre lados opuestos.

    Returns:
        tuple: (es_plausible (bool), diagnóstico (dict con cosenos y ratios reales,
            para poder recalibrar con datos concretos del video)).
    """
    i = half_quad.index(id_missing)
    p_prev = visible_points[half_quad[(i - 1) % 4]]
    p_next = visible_points[half_quad[(i + 1) % 4]]
    p_opp = visible_points[half_quad[(i + 2) % 4]]

    vec_est_1 = p_est - p_prev
    vec_ref_1 = p_opp - p_next
    n1, m1 = np.linalg.norm(vec_est_1), np.linalg.norm(vec_ref_1)
    vec_est_2 = p_next - p_est
    vec_ref_2 = p_prev - p_opp
    n2, m2 = np.linalg.norm(vec_est_2), np.linalg.norm(vec_ref_2)

    if n1 == 0 or m1 == 0 or n2 == 0 or m2 == 0:
        return False, {"error": "vector de longitud cero"}

    cosine_1 = np.dot(vec_est_1, vec_ref_1) / (n1 * m1)
    cosine_2 = np.dot(vec_est_2, vec_ref_2) / (n2 * m2)
    ratio_1 = n1 / m1
    ratio_2 = n2 / m2

    diagnostics = {
        "coseno_1": round(float(cosine_1), 3), "ratio_1": round(float(ratio_1), 3),
        "coseno_2": round(float(cosine_2), 3), "ratio_2": round(float(ratio_2), 3),
    }

    is_plausible = (
        cosine_1 <= -cos_tolerance and ratio_min <= ratio_1 <= ratio_max
        and cosine_2 <= -cos_tolerance and ratio_min <= ratio_2 <= ratio_max
    )
    return is_plausible, diagnostics


def get_footvolley_homography(kpts, confs, img_w, img_h, conf_thresh=0.15, H_last=None):
    """
    Calcula la homografía imagen -> cancha de un frame a partir de sus keypoints.
    Caso A (4+ puntos): homografía directa. Caso B (3 puntos): estima el 4to
    con movimiento rígido desde H_last.

    Args:
        kpts (np.ndarray | None): keypoints (6, 2) ya reescalados a la resolución
            original (ver predict_court_keypoints) -no lee un objeto 'results' crudo,
            para que no pueda colarse un mismatch de espacio de coordenadas.
        confs (np.ndarray): confianza de cada keypoint.
        img_w (int): ancho REAL del frame (se propaga a validate_court_geometry).
        img_h (int): alto REAL del frame.
        conf_thresh (float): confianza mínima para considerar un keypoint visible.
        H_last (np.ndarray | None): homografía de referencia para el Caso B.

    Returns:
        tuple: (H (np.ndarray 3x3) o None, mensaje de estado (str)).
    """
    if kpts is None:
        return None, "No se detectaron keypoints"

    visible_points = {}
    point_confs = {}
    kp_names = ["esquina_sup_izq", "red_sup", "esquina_sup_der", "esquina_inf_der", "red_inf", "esquina_inf_izq"]

    for idx, (pt, conf) in enumerate(zip(kpts, confs)):
        if conf >= conf_thresh:
            visible_points[kp_names[idx]] = pt
            point_confs[kp_names[idx]] = conf

    # CASO A: 4 o más puntos detectados
    if len(visible_points) >= 4:
        src_pts = []
        dst_pts = []
        for expected_name in kp_names:
            if expected_name in visible_points:
                src_pts.append(visible_points[expected_name])
                dst_pts.append(REAL_COURT_MAP[expected_name])

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if validate_court_geometry(H, img_w, img_h):
            return H, f"Homografía directa calculada con {len(visible_points)} puntos"
        else:
            if H_last is not None:
                top3 = sorted(point_confs.items(), key=lambda x: x[1], reverse=True)[:3]
                reduced_points = {k: visible_points[k] for k, v in top3}
                visible_points = reduced_points
            else:
                return None, "Geometría inválida (Caso A) sin fallback posible"

    # CASO B: 3 puntos detectados (estabilización con movimiento rígido parcial)
    if len(visible_points) == 3 and H_last is not None:
        quad = detect_quadrilateral(visible_points)

        if quad is not None:
            id_missing = next(n for n in quad if n not in visible_points)
            p4_est = estimate_point_with_motion(id_missing, visible_points, H_last)

            if p4_est is not None:
                is_plausible, diag = _estimate_is_plausible(quad, visible_points, id_missing, p4_est)
                if is_plausible:
                    src_pts = np.array(
                        [visible_points[n] if n in visible_points else p4_est for n in quad],
                        dtype=np.float32,
                    )
                    dst_pts = np.array([REAL_COURT_MAP[n] for n in quad], dtype=np.float32)

                    H, _ = cv2.findHomography(src_pts, dst_pts, 0)

                    if validate_quadrilateral_geometry(H, quad, img_w, img_h):
                        kind = "esquinas completas" if quad is FULL_CORNERS else "media cancha"
                        return H, f"Homografía resuelta ({kind}): estimando {id_missing} con movimiento rígido -diagnóstico: {diag}"

                return None, f"Estimación de {id_missing} no plausible (Caso B) -diagnóstico: {diag}"

            return None, f"No se pudo estimar {id_missing} con movimiento rígido (Caso B)"

        return None, "Los 3 puntos visibles no encajan en una mitad ni en las esquinas completas (Caso B)"

    return None, f"Puntos insuficientes ({len(visible_points)} detectados) o H_last no disponible"


# =====================================================================
# LOOP PRINCIPAL: calcula homographies_per_frame = {frame_count: H}
# =====================================================================

def compute_homographies(video_path, court_kp_model, conf_thresh=0.35, verbose=False, H_seed_manual=None):
    """
    Loop principal: calcula la homografía de cada frame del video (pooled por
    tipo de plano, continuidad hacia adelante y bootstrap hacia atrás).

    Args:
        video_path (str): ruta del video.
        court_kp_model: modelo YOLO de pose de la cancha.
        conf_thresh (float): confianza mínima para considerar un keypoint visible.
        verbose (bool): imprime el estado de cada frame no resuelto directamente.
        H_seed_manual (np.ndarray | None): homografía opcional calculada A MANO una
            sola vez (por ejemplo, clickeando los 4 puntos conocidos en un frame
            representativo y usando cv2.getPerspectiveTransform). Si el video NUNCA
            tiene un frame con 4+ keypoints confiables (Caso A jamás se dispara),
            esta semilla permite bootstrapear el Caso B desde el frame 0.

    Returns:
        dict: {frame: H (np.ndarray 3x3)} para los frames con homografía.
    """
    cap = cv2.VideoCapture(video_path)

    # Dimensiones REALES del video, no un default fijo
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución del video: {img_w}x{img_h}")

    # --- Pasada 1: recolectar keypoints de TODOS los frames (una sola
    # corrida del modelo, sin calcular homografías todavía) ---
    kpts_per_frame = {}
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        kpts, confs = predict_court_keypoints(court_kp_model, frame, input_size=1280, conf=0.25, iou=0.5)
        if kpts is not None:
            kpts_per_frame[frame_count] = (kpts, confs)
        frame_count += 1
    cap.release()
    print(f"Frames procesados: {frame_count}. Frames con keypoints detectados: {len(kpts_per_frame)}")

    # --- Homografías pooled por tipo de plano (ver sección de arriba) ---
    pooled_homographies = compute_pooled_homographies(kpts_per_frame, img_w, img_h, conf_thresh)
    print(f"Tipos de plano pooled (fijos, con suficientes frames y baja dispersión): {len(pooled_homographies)}")

    # --- Pasada 2 (hacia adelante): igual que antes, con continuidad
    # de H_last frame a frame. Si se proveyó una semilla manual, arranca
    # de ahí en vez de None. Antes de usar H_last (temporal), primero
    # se prueba si hay una homografía pooled compatible con los puntos
    # visibles de este frame -preferida por venir de muchos más datos.
    homographies_per_frame = {}
    H_last = H_seed_manual
    for f in range(frame_count):
        if f in kpts_per_frame:
            kpts, confs = kpts_per_frame[f]
            visible_names = visible_keypoints(confs, conf_thresh)
            H_pooled = find_compatible_pooled_homography(visible_names, pooled_homographies)
            H_reference = H_pooled if H_pooled is not None else H_last

            H, status = get_footvolley_homography(
                kpts, confs, img_w=img_w, img_h=img_h, conf_thresh=conf_thresh, H_last=H_reference
            )
            if verbose and "directa calculada" not in status:
                ref_source = " [semilla: pooled]" if H_pooled is not None else ""
                print(f"Frame {f}{ref_source}: {status}")
            if H is not None:
                H_last = H
        if H_last is not None:
            homographies_per_frame[f] = H_last

    # --- Pasada 3 (bootstrap hacia atrás): si el tramo INICIAL del
    # video nunca tuvo un Caso A exitoso (4+ puntos), H_last se queda
    # en None todo ese tramo y el Caso B nunca puede arrancar -aunque
    # los 3 puntos de esos frames sean perfectos, no hay ninguna
    # referencia de la que partir. Como ya procesamos el video
    # completo, sabemos cuál es la primera homografía que sí se logró
    # más adelante -se usa como semilla y se reprocesa ese tramo
    # inicial CAMINANDO HACIA ATRÁS, exactamente con la misma lógica
    # de Caso B (movimiento rígido + chequeo de plausibilidad).
    #
    # Si muchos frames seguidos de ese tramo tampoco resuelven Caso A/B
    # (por ejemplo, menos de 3 puntos confiables), la semilla se
    # arrastra hacia atrás SIN actualizarse -cada vez más lejos, en
    # frames de distancia, del punto donde se calculó de verdad. Más
    # allá de MAX_BOOTSTRAP_DRAG_FRAMES, es preferible dejar el
    # frame SIN homografía que asignarle una claramente desactualizada.
    first_frame_with_h = min(homographies_per_frame) if homographies_per_frame else None
    if first_frame_with_h is not None and first_frame_with_h > 0:
        print(f"Bootstrap hacia atrás: reprocesando frames 0-{first_frame_with_h - 1} "
              f"(sin ningún Caso A exitoso) usando como semilla la homografía del frame {first_frame_with_h}")
        H_seed = homographies_per_frame[first_frame_with_h]
        frames_since_update = 0
        for f in range(first_frame_with_h - 1, -1, -1):
            if f in kpts_per_frame:
                kpts, confs = kpts_per_frame[f]
                visible_names = visible_keypoints(confs, conf_thresh)
                H_pooled = find_compatible_pooled_homography(visible_names, pooled_homographies)
                H_reference = H_pooled if H_pooled is not None else H_seed

                H, status = get_footvolley_homography(
                    kpts, confs, img_w=img_w, img_h=img_h, conf_thresh=conf_thresh, H_last=H_reference
                )
                if verbose:
                    ref_source = " [semilla: pooled]" if H_pooled is not None else ""
                    print(f"Frame {f}{ref_source} (bootstrap hacia atrás, semilla a {frames_since_update} "
                          f"frames de la última actualización): {status}")
                if H is not None:
                    H_seed = H
                    frames_since_update = 0
                else:
                    frames_since_update += 1
            else:
                frames_since_update += 1
            # Se arrastra H_seed hacia atrás igual que H_last se arrastra
            # hacia adelante -pero solo hasta MAX_BOOTSTRAP_DRAG_FRAMES
            # frames desde la última vez que se actualizó de verdad. Pasado
            # ese límite, el frame queda simplemente SIN homografía -es
            # preferible eso a asignarle una que ya sabemos arrastra
            # demasiado error acumulado.
            if frames_since_update <= MAX_BOOTSTRAP_DRAG_FRAMES:
                homographies_per_frame[f] = H_seed

    print(f"Homografías disponibles: {len(homographies_per_frame)} / {frame_count}")
    return homographies_per_frame


# Uso:
# homographies_per_frame = compute_homographies("video_plano_general.mp4", court_kp_model, conf_thresh=0.35)
#
# Diagnóstico de rachas sin actualizar (recomendado correr después):
# racha_actual = peor_racha = 0
# h_anterior = None
# for f in sorted(homographies_per_frame):
#     h = homographies_per_frame[f]
#     if h is h_anterior:
#         racha_actual += 1
#     else:
#         peor_racha = max(peor_racha, racha_actual)
#         racha_actual = 1
#     h_anterior = h
# peor_racha = max(peor_racha, racha_actual)
# print(f"Racha más larga sin actualizar homografía: {peor_racha} frames")