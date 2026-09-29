-- ============================================================
-- Esquema PostgreSQL - Futevôlei: jugadas y estadísticas
-- Guarda la salida del pipeline (paquete futevolei):
--   meta.csv           -> videos
--   stage2_statistics  -> teams, rallies ("jugadas"), statistics ("resumen")
-- Los comentarios "<- columna" indican de qué columna del DataFrame sale cada dato.
-- ============================================================

-- Para volver a crear todo desde cero (BORRA LOS DATOS), descomentar:
-- DROP TABLE IF EXISTS statistics, rallies, teams, videos CASCADE;
-- DROP TYPE IF EXISTS court_side, shot_result;


-- ------------------------------------------------------------
-- Tipos enumerados: validan a nivel de BD los valores que produce el pipeline
-- ------------------------------------------------------------
CREATE TYPE court_side      AS ENUM ('izq', 'der');
CREATE TYPE shot_result     AS ENUM ('punto', 'defendido', 'fuera', 'desconocido');


-- ------------------------------------------------------------
-- Tabla: videos
-- Un registro por cada video/partido procesado (sale de meta.csv).
-- ------------------------------------------------------------
CREATE TABLE videos (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,                  -- ej. 'set_1'
    source_path     TEXT,                           -- ruta del video original
    fps             NUMERIC(6, 2),                  -- <- meta.fps
    width           INTEGER,                        -- <- meta.width
    height          INTEGER,                        -- <- meta.height
    total_frames    INTEGER,                        -- <- meta.n_frames
    model_version   TEXT,                           -- ej. 'ball_player_yolov8s_best'
    processed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ------------------------------------------------------------
-- Tabla: teams
-- Un registro por equipo (pareja). El nombre es único, así un mismo equipo
-- que aparece en varios videos es una sola fila y se pueden sumar sus
-- estadísticas entre partidos. Viene de `names` en el notebook.
-- ------------------------------------------------------------
CREATE TABLE teams (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,           -- ej. 'Davi e Jamel'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ------------------------------------------------------------
-- Tabla: rallies  (DataFrame "jugadas" de stage2_statistics)
-- Una fila por jugada: quién sacó, cómo terminó y quién ganó el punto.
-- serve_frame / end_frame permiten ir al video a revisar cada jugada.
-- Los equipos se guardan como team_id (se buscan por nombre en teams).
-- ------------------------------------------------------------
CREATE TABLE rallies (
    id                  SERIAL PRIMARY KEY,
    video_id            INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    rally_number        INTEGER NOT NULL,           -- <- rally_id (1, 2, 3...)
    serve_frame         INTEGER NOT NULL,           -- <- serve_frame
    end_frame           INTEGER NOT NULL,           -- <- end_frame
    end_reason          TEXT,                       -- <- end_reason (ground, caught, out_of_frame, lost, manual...)

    -- Saque
    serving_team_id     INTEGER REFERENCES teams(id),   -- <- serving_team
    serve_side          court_side,                 -- <- serve_side
    n_serves            INTEGER,                    -- <- n_serves
    serve_result        shot_result,                -- <- serve_result

    -- Desarrollo
    n_attacks           INTEGER,                    -- <- n_attacks
    attacks_defended    INTEGER,                    -- <- attacks_defended
    n_touches           INTEGER,                    -- <- n_touches
    n_crossings         INTEGER,                    -- <- n_crossings

    -- Final y ganador
    ends_with           TEXT,                       -- <- ends_with (ej. 'ataque Davi e Jamel')
    final_result        shot_result,                -- <- final_result
    winner_team_id      INTEGER REFERENCES teams(id),   -- <- winner_team
    winner_side         court_side,                 -- <- winner_side
    winner_method       TEXT,                       -- <- winner_method (cae_dentro, cae_fuera, saque_siguiente, estimado_...)
    landing_x           REAL,                       -- <- landing_x (cancha 1800x900, NULL si no se vio la caída)
    landing_y           REAL,                       -- <- landing_y

    -- Lados de cada equipo en esta jugada
    team_left_id        INTEGER REFERENCES teams(id),   -- <- team_left
    team_right_id       INTEGER REFERENCES teams(id),   -- <- team_right
    side_switch         BOOLEAN,                    -- <- side_switch (primera jugada con los lados cambiados)

    -- Calidad: fracción de frames con referencia para ubicar la red (0-1)
    coverage_pan        REAL,                       -- <- coverage_pan
    coverage_h          REAL,                       -- <- coverage_h
    coverage_players    REAL,                       -- <- coverage_players

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, rally_number),
    CHECK (end_frame >= serve_frame)
);


-- ------------------------------------------------------------
-- Tabla: statistics  (DataFrame "resumen" de stage2_statistics)
-- Una fila por equipo y video.
-- ------------------------------------------------------------
CREATE TABLE statistics (
    id                          SERIAL PRIMARY KEY,
    video_id                    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    team_id                     INTEGER NOT NULL REFERENCES teams(id),   -- <- team
    points                      INTEGER,            -- <- points
    serves                      INTEGER,            -- <- serves
    aces                        INTEGER,            -- <- aces
    serves_defended             INTEGER,            -- <- serves_defended
    serves_out                  INTEGER,            -- <- serves_out
    attacks                     INTEGER,            -- <- attacks
    attacks_won                 INTEGER,            -- <- attacks_won
    attacks_defended            INTEGER,            -- <- attacks_defended
    attacks_out                 INTEGER,            -- <- attacks_out
    attack_efficiency_pct       NUMERIC(5, 1),      -- <- attack_efficiency_pct (NULL si no hubo ataques)
    opponent_attacks_defended   INTEGER,            -- <- opponent_attacks_defended
    processed_at                TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, team_id)
);


-- (No hacen falta índices extra: UNIQUE (video_id, rally_number) y
--  UNIQUE (video_id, team_id) ya crean índices para buscar por video.)


-- ============================================================
-- Consultas de ejemplo
-- ============================================================

-- Resumen por equipo de un video
-- SELECT t.name, s.*
-- FROM statistics s JOIN teams t ON t.id = s.team_id
-- WHERE s.video_id = 1;

-- Marcador acumulado jugada a jugada
-- SELECT r.rally_number, w.name AS ganador,
--        sum((r.winner_team_id = t.id)::int) OVER (PARTITION BY t.id ORDER BY r.rally_number) AS puntos_acum,
--        t.name AS equipo
-- FROM rallies r
-- JOIN teams w ON w.id = r.winner_team_id
-- JOIN teams t ON t.id IN (r.team_left_id, r.team_right_id)
-- WHERE r.video_id = 1
-- ORDER BY r.rally_number, t.name;

-- Jugadas donde el ganador no salió de ver la caída (para revisarlas en el video)
-- SELECT rally_number, serve_frame, end_frame, winner_method
-- FROM rallies
-- WHERE video_id = 1 AND winner_method NOT IN ('cae_dentro', 'cae_fuera')
-- ORDER BY rally_number;

-- Puntos ganados con saque propio vs. saque rival
-- SELECT t.name,
--        count(*) FILTER (WHERE r.serving_team_id = t.id)  AS con_saque_propio,
--        count(*) FILTER (WHERE r.serving_team_id <> t.id) AS con_saque_rival
-- FROM rallies r JOIN teams t ON t.id = r.winner_team_id
-- WHERE r.video_id = 1
-- GROUP BY t.name;
