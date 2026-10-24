-- ClimateLens database schema (PostgreSQL)
--
-- Responses are anonymous: no employee identifiers are stored.

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------

CREATE TABLE departments (
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,                 -- e.g. 'Production'
    is_active   BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE categories (
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code        TEXT    NOT NULL UNIQUE,                 -- e.g. 'workload'
    is_active   BOOLEAN NOT NULL DEFAULT TRUE
);

-- ---------------------------------------------------------------------------
-- Survey input
-- ---------------------------------------------------------------------------

-- One uploaded CSV; its upload month is the month its responses count towards
CREATE TABLE survey_batches (
    id          INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        TEXT        NOT NULL,                    -- e.g. 'Monthly survey 2026-10'
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    row_count   INTEGER     NOT NULL DEFAULT 0 CHECK (row_count >= 0)
);

CREATE TABLE responses (
    id              INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id        INTEGER     NOT NULL REFERENCES survey_batches(id) ON DELETE CASCADE,
    department_id   INTEGER     REFERENCES departments(id),  -- NULL when the CSV has no department
    feedback_text   TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_survey_batches_uploaded_at ON survey_batches (uploaded_at);
CREATE INDEX idx_responses_batch_dept       ON responses (batch_id, department_id);

-- ---------------------------------------------------------------------------
-- Predictions
-- ---------------------------------------------------------------------------

-- One sentiment result per response
CREATE TABLE sentiment_predictions (
    id           INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    response_id  INTEGER     NOT NULL UNIQUE REFERENCES responses(id) ON DELETE CASCADE,
    label        TEXT        NOT NULL CHECK (label IN ('Positive', 'Neutral', 'Negative')),
    confidence   REAL        CHECK (confidence BETWEEN 0 AND 1),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One category per response
CREATE TABLE category_predictions (
    id           INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    response_id  INTEGER     NOT NULL UNIQUE REFERENCES responses(id) ON DELETE CASCADE,
    category_id  INTEGER     NOT NULL REFERENCES categories(id),
    confidence   REAL        CHECK (confidence BETWEEN 0 AND 1),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_category_predictions_category ON category_predictions (category_id);

-- ---------------------------------------------------------------------------
-- Views used by the dashboard
-- ---------------------------------------------------------------------------

-- One row per response with its predictions (same shape as the dashboard DataFrame)
CREATE VIEW v_analyzed_responses AS
SELECT
    r.id                                 AS response_id,
    to_char(b.uploaded_at, 'YYYY-MM')    AS month,
    COALESCE(d.name, 'Unknown')          AS department,
    r.feedback_text                      AS feedback,
    sp.label                             AS sentiment,
    c.code                               AS category,
    sp.confidence                        AS confidence
FROM responses r
JOIN survey_batches b ON b.id = r.batch_id
LEFT JOIN departments d ON d.id = r.department_id
JOIN sentiment_predictions sp ON sp.response_id = r.id
LEFT JOIN category_predictions cp ON cp.response_id = r.id
LEFT JOIN categories c ON c.id = cp.category_id;

-- Climate Index = 50 + 50 * (positive - negative) / responses  (0..100, 50 = balanced)
CREATE VIEW v_climate_monthly AS
SELECT
    month,
    department,
    category,
    COUNT(*)                                       AS responses,
    COUNT(*) FILTER (WHERE sentiment = 'Positive') AS positive,
    COUNT(*) FILTER (WHERE sentiment = 'Neutral')  AS neutral,
    COUNT(*) FILTER (WHERE sentiment = 'Negative') AS negative,
    50 + 50.0 * (COUNT(*) FILTER (WHERE sentiment = 'Positive')
               - COUNT(*) FILTER (WHERE sentiment = 'Negative')) / COUNT(*) AS climate_index
FROM v_analyzed_responses
GROUP BY month, department, category;
