-- ClimateLens: count each response towards its survey date, not the upload day
--
-- Before this, a response's month came from survey_batches.uploaded_at, so every batch
-- saved this month landed in this month, whatever dates the CSV carried.
-- responses.survey_date keeps the CSV `date`; rows without one (including everything
-- saved before this migration) fall back to the upload day as before.
--
-- Run once in the Supabase SQL Editor, after 20261025000000_auth_rls.sql.

BEGIN;

ALTER TABLE responses ADD COLUMN survey_date DATE;

CREATE INDEX idx_responses_survey_date ON responses (survey_date);

-- Same columns as before; only month and date change where they come from.
-- WITH (security_invoker) is repeated so the replaced view keeps applying the caller's RLS.
CREATE OR REPLACE VIEW v_analyzed_responses WITH (security_invoker = true) AS
SELECT
    r.id                                                          AS response_id,
    to_char(COALESCE(r.survey_date, b.uploaded_at::date), 'YYYY-MM') AS month,
    COALESCE(d.name, 'Unknown')                                   AS department,
    r.feedback_text                                               AS feedback,
    sp.label                                                      AS sentiment,
    c.code                                                        AS category,
    sp.confidence                                                 AS confidence,
    COALESCE(r.survey_date, b.uploaded_at::date)                  AS date
FROM responses r
JOIN survey_batches b ON b.id = r.batch_id
LEFT JOIN departments d ON d.id = r.department_id
JOIN sentiment_predictions sp ON sp.response_id = r.id
LEFT JOIN category_predictions cp ON cp.response_id = r.id
LEFT JOIN categories c ON c.id = cp.category_id;

COMMIT;

-- Let the API see the new column straight away
NOTIFY pgrst, 'reload schema';
