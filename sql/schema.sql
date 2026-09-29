-- ============================================================
-- Esquema PostgreSQL para futevolei pipeline final
-- Proyecto: tracking de pelota + jugadores -> jugadas + estadísticas
-- ============================================================

-- ------------------------------------------------------------
-- Tabla: videos
-- Un registro por cada video/partido procesado por el modelo.
-- ------------------------------------------------------------
CREATE TABLE videos (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    source_path     TEXT,
    fps             NUMERIC(6, 2),
    width           INTEGER,
    height          INTEGER,
    total_frames    INTEGER,
    model_version   TEXT,
    processed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- Tabla: teams
-- ------------------------------------------------------------
CREATE TABLE teams (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    processed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- Tabla: rallies
-- Una fila por cada jugada detectada en el video.
-- Corresponde a e2["rally_stats"] del pipeline.
-- ------------------------------------------------------------
CREATE TABLE rallies (
    id                BIGSERIAL PRIMARY KEY,
    video_id          INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,

    rally_number      INTEGER NOT NULL,
    serve_frame       INTEGER NOT NULL,
    end_frame         INTEGER NOT NULL,
    end_reason        TEXT,

    serve_side        TEXT,
    n_serves          INTEGER,
    serve_result      TEXT,
    n_attacks         INTEGER,
    attacks_team_1    INTEGER,
    attacks_team_2    INTEGER,
    attacks_defended  INTEGER,

    n_touches         INTEGER,
    n_crossings       INTEGER,

    ends_with         TEXT,
    final_result      TEXT,
    winner_side       TEXT,
    winner_method     TEXT,

    landing_x         REAL,
    landing_y         REAL,

    side_switch       BOOLEAN,
    coverage_pan      REAL,
    coverage_h        REAL,
    coverage_players  REAL,

    serving_team_id   INTEGER REFERENCES teams(id),
    winner_team_id    INTEGER REFERENCES teams(id),
    team_left_id      INTEGER REFERENCES teams(id),
    team_right_id     INTEGER REFERENCES teams(id),

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- Tabla: shots - deprecated
-- Una fila por cada saque o ataque dentro de una jugada.
-- Corresponde a e2["shots"] del pipeline.
-- ------------------------------------------------------------
-- CREATE TABLE shots (
--     id                BIGSERIAL PRIMARY KEY,
--     video_id          INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
--     rally_number      INTEGER NOT NULL,
--     frame             INTEGER NOT NULL,
--     team_id           INTEGER REFERENCES teams(id),
--     side              TEXT,
--     shot_type         TEXT NOT NULL,
--     result            TEXT,
--     opponent_touched  BOOLEAN,
--     crossing_seen     BOOLEAN,

--     created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
-- );

-- ------------------------------------------------------------
-- Tabla: statistics
-- Resumen por equipo. Corresponde a e2["summary"] del pipeline.
-- ------------------------------------------------------------
CREATE TABLE statistics (
    id                          SERIAL PRIMARY KEY,
    video_id                    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    team_id                     INTEGER NOT NULL REFERENCES teams(id),
    points                      INTEGER,
    serves                      INTEGER,
    aces                        INTEGER,
    serves_defended             INTEGER,
    serves_out                  INTEGER,
    attacks                     INTEGER,
    attacks_won                 INTEGER,
    attacks_defended            INTEGER,
    attacks_out                 INTEGER,
    attack_efficiency_pct       REAL,
    opponent_attacks_defended   INTEGER,
    processed_at                TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- Índices
-- ------------------------------------------------------------

CREATE INDEX idx_rallies_video ON rallies (video_id);
CREATE INDEX idx_rallies_video_rally ON rallies (video_id, rally_number);

-- CREATE INDEX idx_shots_video ON shots (video_id);
-- CREATE INDEX idx_shots_video_rally ON shots (video_id, rally_number);

CREATE INDEX idx_statistics_video ON statistics (video_id);
CREATE INDEX idx_statistics_video_team ON statistics (video_id, team_id);
