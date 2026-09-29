"""
futevolei: jugadas y estadísticas de futevôlei a partir del video de la transmisión.

Módulos (en el orden en que se usan):

  ETAPA 1 - jugadas
    detection.py          detecciones YOLO (pelota + jugadores) y limpieza (pelotas de repuesto,
                          objetos fijos confundidos con la pelota)
    ball_trajectory.py    una pelota por frame: interpolación, suavizado y velocidades
    broadcast.py          repeticiones (animación de la pelota gigante) y cambios de vista
    rallies.py            segmentación de jugadas: saque -> arena / atrapada / fuera de cuadro
    video_output.py       video de revisión / compilado, video con cajas y gráfico de calibración

  ETAPA 2 - estadísticas
    camera.py             paneo y cortes de la cámara; posición de la red en cada frame
    homographies.py       homografía imagen -> cancha con el modelo de keypoints de la cancha
    rally_homographies.py homografías solo en las jugadas y estabilizadas con el paneo
    court_geometry.py     geometría de la cancha: lados, red en la imagen, imagen -> cancha
    ball_sides.py         toques, lado de la pelota en cada frame y cruces de la red
    match_stats.py        ganador de cada punto, saques / ataques con su resultado y resumen

  Comunes
    common.py             metadatos del video y utilidades
    pipeline.py           stage1_detect y stage2_statistics (lo que se llama desde el notebook)
    database.py           guardar la salida de la etapa 2 en PostgreSQL (save_to_sql)

Uso:
    from futevolei import stage1_detect, stage2_statistics
"""
from .pipeline import stage1_detect, stage2_statistics

__all__ = ["stage1_detect", "stage2_statistics"]
