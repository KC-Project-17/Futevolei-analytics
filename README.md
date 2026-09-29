# Futevôlei CV: jugadas y estadísticas desde el video

Proyecto universitario de visión por computador. A partir del video de la transmisión de un
partido de futevôlei detecta cada jugada y calcula, sin mirar el marcador:

- quién saca y cómo termina el saque,
- los ataques de cada equipo y su resultado (punto / defendido / fuera),
- quién gana cada punto,
- un resumen por equipo (puntos, saques, aces, ataques, eficacia).

## Estructura

```
futevolei_code/          paquete de Python (importado desde el notebook)
  detection.py           detecciones YOLO de pelota y jugadores + limpieza
  ball_trajectory.py     una pelota por frame: interpolación, suavizado, velocidades
  broadcast.py           repeticiones y cambios de vista de la transmisión
  rallies.py             segmentación de jugadas (saque -> fin)
  video_output.py        video de revisión, video con cajas, gráfico de calibración
  camera.py              paneo de la cámara y posición de la red en cada frame
  homographies.py        homografía imagen -> cancha con el modelo de keypoints
  rally_homographies.py  homografías solo en las jugadas, estabilizadas con el paneo
  court_geometry.py      geometría de la cancha
  ball_sides.py          toques, lado de la pelota y cruces de la red
  match_stats.py         ganador de cada punto, saques / ataques y resumen
  common.py              utilidades (metadatos del video)
  pipeline.py            stage1_detect y stage2_statistics
  database.py            guardar los resultados en PostgreSQL
notebooks/
  futevolei_pipeline.ipynb   notebook de Colab con todo el flujo
sql/
  schema.sql             tablas de PostgreSQL (videos, teams, rallies, statistics)
```

Los modelos (`.pt`), los videos y los resultados **no** están en el repositorio: quedan en Google Drive.

## Uso (Google Colab)

1. Abrir `notebooks/futevolei_pipeline.ipynb` en Colab (con GPU).
2. Importar el código desde GitHub (opción A) o desde Drive (opción B).
3. **Paso 1** `stage1_detect`: detecciones, jugadas y `review_video.mp4`.
4. **Revisión manual** mirando ese video: jugadas falsas (`remove`), faltantes (`add`),
   primera jugada después de cada cambio de lado (`side_switches`) y nombres de los equipos (`names`).
5. **Paso 2** `stage2_statistics`: estadísticas (`rally_stats`, `shots`, `summary`) y video con cajas.
6. (Opcional) guardar en PostgreSQL con `save_to_sql`.

```python
from futevolei_code import stage1_detect, stage2_statistics

e1 = stage1_detect(VIDEO, ball_player_model, out_dir=OUT_FOLDER)
e2 = stage2_statistics(VIDEO, e1, court_kp_model, out_dir=OUT_FOLDER,
                       remove=[], add=[], side_switches=[7, 13, 19, 25, 31, 37],
                       names={"Equipo 1": "...", "Equipo 2": "..."})
e2["summary"]
```

## Base de datos

`sql/schema.sql` crea las tablas en PostgreSQL (probado con Supabase: correrlo en el *SQL Editor*).
Desde Colab conectarse con el **Session pooler** de Supabase (Colab no soporta IPv6).

## Requisitos

```
pip install -r requirements.txt
```
