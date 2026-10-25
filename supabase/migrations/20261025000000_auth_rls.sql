-- ClimateLens: per-user data ownership (Supabase Auth + Row Level Security)
--
-- Every survey batch, response and department belongs to the user who uploaded it.
-- RLS policies make each logged-in user see and change only their own rows;
-- the dashboard must therefore query with the user's session, not the secret key.
--
-- Run once in the Supabase SQL Editor, after supabase/schema.sql.
-- If the database already has rows, create the account that should own them first
-- (Authentication -> Users -> Add user) and put its email in owner_email below.

BEGIN;

-- ---------------------------------------------------------------------------
-- Ownership columns
-- ---------------------------------------------------------------------------

ALTER TABLE survey_batches ADD COLUMN owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE responses      ADD COLUMN owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE departments    ADD COLUMN owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

-- Rows saved before this migration have no owner; give them to one account
DO $$
DECLARE
    owner_email TEXT := 'usera@test.com';  -- EDIT before running
    first_owner UUID;
BEGIN
    IF EXISTS (SELECT 1 FROM survey_batches) OR EXISTS (SELECT 1 FROM departments) THEN
        SELECT id INTO first_owner FROM auth.users WHERE email = owner_email;
        IF first_owner IS NULL THEN
            RAISE EXCEPTION 'No auth user with email %. Create it under Authentication -> Users, then set owner_email.', owner_email;
        END IF;

        UPDATE survey_batches SET owner_id = first_owner;
        UPDATE responses r SET owner_id = b.owner_id FROM survey_batches b WHERE b.id = r.batch_id;
        UPDATE departments SET owner_id = first_owner;
    END IF;
END $$;

-- New rows are owned by whoever inserts them
ALTER TABLE survey_batches ALTER COLUMN owner_id SET NOT NULL, ALTER COLUMN owner_id SET DEFAULT auth.uid();
ALTER TABLE responses      ALTER COLUMN owner_id SET NOT NULL, ALTER COLUMN owner_id SET DEFAULT auth.uid();
ALTER TABLE departments    ALTER COLUMN owner_id SET NOT NULL, ALTER COLUMN owner_id SET DEFAULT auth.uid();

-- Two users may both have an 'IT' department; names are unique per owner
ALTER TABLE departments DROP CONSTRAINT departments_name_key;
ALTER TABLE departments ADD CONSTRAINT departments_owner_name_key UNIQUE (owner_id, name);

CREATE INDEX idx_survey_batches_owner ON survey_batches (owner_id);
CREATE INDEX idx_responses_owner      ON responses (owner_id);

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------

ALTER TABLE departments           ENABLE ROW LEVEL SECURITY;
ALTER TABLE categories            ENABLE ROW LEVEL SECURITY;
ALTER TABLE survey_batches        ENABLE ROW LEVEL SECURITY;
ALTER TABLE responses             ENABLE ROW LEVEL SECURITY;
ALTER TABLE sentiment_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE category_predictions  ENABLE ROW LEVEL SECURITY;

-- (SELECT auth.uid()) is evaluated once per query instead of once per row
CREATE POLICY "Own batches" ON survey_batches FOR ALL TO authenticated
    USING (owner_id = (SELECT auth.uid()))
    WITH CHECK (owner_id = (SELECT auth.uid()));

CREATE POLICY "Own departments" ON departments FOR ALL TO authenticated
    USING (owner_id = (SELECT auth.uid()))
    WITH CHECK (owner_id = (SELECT auth.uid()));

-- A response must sit in the user's own batch and point at the user's own department
CREATE POLICY "Own responses" ON responses FOR ALL TO authenticated
    USING (owner_id = (SELECT auth.uid()))
    WITH CHECK (
        owner_id = (SELECT auth.uid())
        AND EXISTS (SELECT 1 FROM survey_batches b
                    WHERE b.id = batch_id AND b.owner_id = (SELECT auth.uid()))
        AND (department_id IS NULL
             OR EXISTS (SELECT 1 FROM departments d
                        WHERE d.id = department_id AND d.owner_id = (SELECT auth.uid())))
    );

-- Predictions belong to whoever owns the response
CREATE POLICY "Own sentiment predictions" ON sentiment_predictions FOR ALL TO authenticated
    USING (EXISTS (SELECT 1 FROM responses r
                   WHERE r.id = response_id AND r.owner_id = (SELECT auth.uid())))
    WITH CHECK (EXISTS (SELECT 1 FROM responses r
                        WHERE r.id = response_id AND r.owner_id = (SELECT auth.uid())));

CREATE POLICY "Own category predictions" ON category_predictions FOR ALL TO authenticated
    USING (EXISTS (SELECT 1 FROM responses r
                   WHERE r.id = response_id AND r.owner_id = (SELECT auth.uid())))
    WITH CHECK (EXISTS (SELECT 1 FROM responses r
                        WHERE r.id = response_id AND r.owner_id = (SELECT auth.uid())));

-- Categories are the model's shared label set: everyone reads, anyone may add, nobody edits
CREATE POLICY "Read categories" ON categories FOR SELECT TO authenticated USING (true);
CREATE POLICY "Add categories"  ON categories FOR INSERT TO authenticated WITH CHECK (true);

-- ---------------------------------------------------------------------------
-- Grants: logged-in users only; nothing for anonymous visitors
-- ---------------------------------------------------------------------------

REVOKE ALL ON departments, categories, survey_batches, responses,
              sentiment_predictions, category_predictions,
              v_analyzed_responses, v_climate_monthly
FROM anon;

GRANT SELECT, INSERT, UPDATE, DELETE
    ON departments, survey_batches, responses, sentiment_predictions, category_predictions
    TO authenticated;
-- UPDATE (code) is only for the upsert's conflict path; no policy allows an actual edit
GRANT SELECT, INSERT, UPDATE (code) ON categories TO authenticated;
GRANT SELECT ON v_analyzed_responses, v_climate_monthly TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Same columns as schema.sql plus `date` (the batch's upload day) for the dashboard tables
CREATE OR REPLACE VIEW v_analyzed_responses AS
SELECT
    r.id                                 AS response_id,
    to_char(b.uploaded_at, 'YYYY-MM')    AS month,
    COALESCE(d.name, 'Unknown')          AS department,
    r.feedback_text                      AS feedback,
    sp.label                             AS sentiment,
    c.code                               AS category,
    sp.confidence                        AS confidence,
    b.uploaded_at::date                  AS date
FROM responses r
JOIN survey_batches b ON b.id = r.batch_id
LEFT JOIN departments d ON d.id = r.department_id
JOIN sentiment_predictions sp ON sp.response_id = r.id
LEFT JOIN category_predictions cp ON cp.response_id = r.id
LEFT JOIN categories c ON c.id = cp.category_id;

-- By default a view runs with its creator's rights and skips RLS, which would
-- show every user's rows. security_invoker makes it apply the caller's policies.
ALTER VIEW v_analyzed_responses SET (security_invoker = true);
ALTER VIEW v_climate_monthly    SET (security_invoker = true);

COMMIT;
