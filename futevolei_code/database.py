"""
Guardar los resultados del pipeline en PostgreSQL (esquema schema.sql):

    videos      <- meta (meta.csv)
    teams       <- nombres de los equipos (los de `names` en el notebook)
    rallies     <- e2["rally_stats"]
    shots       <- e2["shots"]
    statistics  <- e2["summary"]

Todo se inserta en UNA transacción: si algo falla, no queda nada a medias.
Si el video ya estaba en la base (mismo nombre), se reemplaza.
"""
import numpy as np
import pandas as pd

RALLY_COLUMNS = [
    ("rally_number", "rally_id"), ("serve_frame", "serve_frame"), ("end_frame", "end_frame"),
    ("end_reason", "end_reason"), ("serve_side", "serve_side"), ("n_serves", "n_serves"),
    ("serve_result", "serve_result"), ("n_attacks", "n_attacks"),
    ("attacks_team_1", "attacks_team_1"), ("attacks_team_2", "attacks_team_2"),
    ("attacks_defended", "attacks_defended"), ("n_touches", "n_touches"),
    ("n_crossings", "n_crossings"), ("ends_with", "ends_with"),
    ("final_result", "final_result"), ("winner_side", "winner_side"),
    ("winner_method", "winner_method"), ("landing_x", "landing_x"),
    ("landing_y", "landing_y"), ("side_switch", "side_switch"),
    ("coverage_pan", "coverage_pan"), ("coverage_h", "coverage_h"),
    ("coverage_players", "coverage_players"),
]
RALLY_TEAM_COLUMNS = [
    ("serving_team_id", "serving_team"), ("winner_team_id", "winner_team"),
    ("team_left_id", "team_left"), ("team_right_id", "team_right"),
]
# SHOT_COLUMNS = [
#     ("rally_number", "rally_id"), ("frame", "frame"), ("side", "side"),
#     ("shot_type", "shot_type"), ("result", "result"),
#     ("opponent_touched", "opponent_touched"), ("crossing_seen", "crossing_seen"),
# ]
STAT_COLUMNS = [
    "points", "serves", "aces", "serves_defended", "serves_out",
    "attacks", "attacks_won", "attacks_defended", "attacks_out",
    "attack_efficiency_pct", "opponent_attacks_defended",
]


def _py(v):
    """
    Convierte un valor de pandas / numpy a un tipo que entiende psycopg2.

    Args:
        v: valor de una celda de un DataFrame.

    Returns:
        None para NaN / NA, int / float / bool / str de Python para el resto.
    """
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.generic):
        v = v.item()
        if isinstance(v, float) and np.isnan(v):
            return None
    return v


def connect(host, dbname, user, password, port=5432, sslmode="prefer"):
    """
    Abre una conexión a PostgreSQL.

    Args:
        host (str): servidor de la base.
        dbname (str): nombre de la base de datos.
        user (str): usuario.
        password (str): contraseña.
        port (int): puerto.
        sslmode (str): 'require' para servicios en la nube (Supabase, Neon, Render...).

    Returns:
        psycopg2.extensions.connection: conexión abierta.
    """
    import psycopg2
    return psycopg2.connect(host=host, dbname=dbname, user=user, password=password,
                            port=port, sslmode=sslmode)


def _team_ids(cur, team_names):
    """
    Inserta los equipos que no existen y devuelve el id de cada uno.

    Args:
        cur: cursor de psycopg2.
        team_names (iterable): nombres de los equipos.

    Returns:
        dict: {nombre: id}.
    """
    ids = {}
    for name in sorted(set(team_names)):
        if not isinstance(name, str):
            continue
        cur.execute("INSERT INTO teams (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (name,))
        cur.execute("SELECT id FROM teams WHERE name = %s", (name,))
        ids[name] = cur.fetchone()[0]
    return ids


def save_to_sql(conn, stage2, meta, video_name, source_path=None, model_version=None,
                replace=True, verbose=True):
    """
    Guarda en la base el resultado de stage2_statistics.

    Args:
        conn: conexión de psycopg2 (ver connect).
        stage2 (dict): el dict que devolvió stage2_statistics (usa "rally_stats",
            "shots" y "summary").
        meta (pd.DataFrame | dict): metadatos del video (e1["meta"] o pd.read_csv(".../meta.csv")).
        video_name (str): nombre del video en la tabla videos (ej. 'set_1'); identifica al video.
        source_path (str | None): ruta del video original.
        model_version (str | None): modelo de pelota y jugadores usado.
        replace (bool): si ya hay un video con ese nombre, se borra (con sus jugadas y
            estadísticas) y se vuelve a insertar. Con False, da error si ya existe.
        verbose (bool): imprime qué se insertó.

    Returns:
        int: id del video en la tabla videos.
    """
    from psycopg2.extras import execute_values

    rallies = stage2["rally_stats"]
    shots = stage2.get("shots", pd.DataFrame())
    summary = stage2["summary"]
    if isinstance(meta, pd.DataFrame):
        meta = meta.iloc[0].to_dict()

    names = set()
    for col in ("serving_team", "winner_team", "team_left", "team_right"):
        if col in rallies.columns:
            names |= set(rallies[col].dropna())
    if "team" in summary.columns:
        names |= set(summary["team"].dropna())
    if "team" in shots.columns:
        names |= set(shots["team"].dropna())
    names = {n for n in names if isinstance(n, str)}

    if verbose and names & {"Equipo 1", "Equipo 2"}:
        print("⚠️  Los equipos se llaman 'Equipo 1' / 'Equipo 2': en la base se van a mezclar con los "
              "de otros videos. Poné los nombres reales en `names` antes de guardar.")

    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM videos WHERE name = %s", (video_name,))
            old = [row[0] for row in cur.fetchall()]
            if old and not replace:
                raise ValueError(f"El video '{video_name}' ya está en la base (id {old}); usá replace=True")
            if old:
                cur.execute("DELETE FROM videos WHERE id = ANY(%s)", (old,))

            cur.execute(
                """INSERT INTO videos (name, source_path, fps, width, height, total_frames,
                                       model_version, processed_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, now()) RETURNING id""",
                (video_name, source_path, _py(meta["fps"]), _py(meta["width"]), _py(meta["height"]),
                 _py(meta["n_frames"]), model_version))
            video_id = cur.fetchone()[0]

            team_id = _team_ids(cur, names)
            tid = lambda name: team_id.get(name) if isinstance(name, str) else None

            cols = ["video_id"] + [c for c, _ in RALLY_COLUMNS] + [c for c, _ in RALLY_TEAM_COLUMNS]
            rows = [[video_id] + [_py(r[src]) for _, src in RALLY_COLUMNS]
                    + [tid(r[src]) for _, src in RALLY_TEAM_COLUMNS]
                    for _, r in rallies.iterrows()]
            if rows:
                execute_values(cur, f"INSERT INTO rallies ({', '.join(cols)}) VALUES %s", rows)

            # if not shots.empty:
            #     shot_cols = ["video_id"] + [c for c, _ in SHOT_COLUMNS] + ["team_id"]
            #     shot_rows = [[video_id] + [_py(r[src]) for _, src in SHOT_COLUMNS]
            #                  + [tid(r["team"])]
            #                  for _, r in shots.iterrows()]
            #     execute_values(cur, f"INSERT INTO shots ({', '.join(shot_cols)}) VALUES %s", shot_rows)

            rows = [[video_id, tid(s["team"])] + [_py(s[c]) for c in STAT_COLUMNS]
                    for _, s in summary.iterrows()]
            if rows:
                template = "(" + ", ".join(["%s"] * len(rows[0])) + ", now())"
                execute_values(cur, f"INSERT INTO statistics (video_id, team_id, {', '.join(STAT_COLUMNS)}, "
                                    f"processed_at) VALUES %s", rows, template=template)

    if verbose:
        print(f"Guardado en la base: video '{video_name}' (id {video_id}), {len(rallies)} jugadas, "
              f"{len(shots)} golpes, "
              f"{len(summary)} equipos ({', '.join(sorted(names))})"
              + (" [reemplazó la versión anterior]" if old else ""))
    return video_id
