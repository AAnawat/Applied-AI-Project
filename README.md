# ClimateLens

Streamlit dashboard for Thai employee feedback, sentiment and organizational climate.

## Setup

Run commands from the repository root:

```bash
conda create -n climatelens python=3.11 pip -y
conda activate climatelens
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

The Upload & Analyze page and prediction command use the exported sentiment model.
Topic categories in the upload page still use temporary keyword matching.
After uploading a CSV, review the results, enter a survey batch name and click
**Save to Supabase**. The page shows the database batch ID on success and disables
repeat saves of the same file/name within that session. Merely selecting a file or
clicking **Use this data in the dashboard** does not save it to the database.
Users sign in with their Supabase account first; the dashboard then shows the responses
saved in that account (**Saved**), or an unsaved upload (**Uploaded**). A browser refresh
starts a new Streamlit session, so it signs the user out.

## Structure

- `dashboard/app.py`: entry point, sign-in gate, navigation and global filters (months, departments, data source)
- `dashboard/auth.py`: Supabase Auth sign-in; one Supabase client per browser session
- `dashboard/views/`: Overview, Departments, Trends, Feedback Explorer, Upload & Analyze
- `dashboard/climate.py`: data access, filters, Climate Index (`50 + 50 × (pos − neg) / n`; ≥60 Healthy, 45–60 Watch, <45 Needs attention)
- `dashboard/mock_data.py`: generated Thai survey responses (Jan–Oct 2026, 7 departments, 7 categories)
- `dashboard/classifier.py`: keyword placeholder; replace `classify()` with the real sentiment model + LLM categorizer

Dashboard pages show an upload prompt when the selected data source has no responses. If filters exclude all responses, clear or broaden the filters. Uploaded CSVs must contain at least one non-empty `feedback` value.

## Model

Place the exported complete scikit-learn Pipeline (fitted TF-IDF + classifier)
at `models/sentiment.joblib`, or pass `--model /path/to/model.joblib`.
Use the same scikit-learn version as the export environment.
Only load trusted joblib files. This project includes prediction only, with no model fitting.

## Database: schema from project/message.txt

The previous single `feedback_predictions` table is no longer used.
The code now follows these tables and views:

| Table/view | Purpose |
| --- | --- |
| `departments` | Unique department names |
| `categories` | Unique category codes |
| `survey_batches` | One upload, server upload time and row count |
| `responses` | Feedback text linked to a batch and optional department |
| `sentiment_predictions` | One sentiment label/probability per response |
| `category_predictions` | Optional category result per response |
| `v_analyzed_responses` | Joined records for the dashboard |
| `v_climate_monthly` | Climate Index grouped by upload month, department, category |

Run `supabase/schema.sql` in your Supabase SQL Editor. It creates the tables/views
from message.txt and backend permissions, including deletion of failed upload batches.
The script can be rerun on matching existing tables to apply permissions. It does not
migrate an incompatible schema or delete existing data.
This is manual setup; Python does not run the SQL automatically.
No additional SQL function or RPC setup is needed.

Then run `supabase/migrations/20261025000000_auth_rls.sql` the same way. It gives
every batch, response and department an `owner_id` (the signed-in user), turns on Row
Level Security so each user reads and writes only their own rows, and makes the views
apply those rules. If the database already has rows, first create the account that
should own them and set `owner_email` at the top of the file.

### Accounts

Create users in Supabase under **Authentication → Users → Add user** (tick
**Auto Confirm User**), and turn off **Allow new users to sign up** so nobody can
create an account through the API.

### Connection settings

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in
the ClimateLens project's URL and **publishable (anon)** key:

```toml
supabase_url = "https://your-project.supabase.co"
supabase_key = "your-publishable-key"
```

The real secrets file is ignored by Git. In Streamlit hosting, set the same TOML
in the app's Secrets settings. Restart after editing. The old `.env` configuration
is not used.

The Secret/service_role key is rejected on purpose: it bypasses Row Level Security,
so the dashboard would show every user's data. Never paste it into chat or commit it.
Each browser session gets its own client (`st.session_state`), never a shared
`@st.cache_resource` one, because the client holds the signed-in user's token.

If saving fails with database code `42501`, the user is not signed in or the
auth/RLS migration has not been run.

## Predict and save

```bash
python -m sentiment.predict --email you@example.com --text "เพื่อนร่วมงานช่วยเหลือดี" --batch-name "Single message"
python -m sentiment.predict --email you@example.com --input /path/to/feedback.csv --batch-name "Monthly survey 2026-10"
```

The command asks for that account's password and saves into its account.

If omitted, the batch name defaults to the input filename without its extension,
or "Single feedback" for a text message. Batch names do not control the upload month.

To preview without any database connection or write:

```bash
python -m sentiment.predict --text "วันนี้มีประชุม" --dry-run
```

Input CSV requires `feedback`. Optional `department` maps to a department reference.
The prediction command appends sentiment and confidence; input must not already
contain those columns.

Optional `category` must be the category code to store (for example `workload`).
It is preserved if already supplied by a categorizer; no category is invented by
the sentiment model. `category_confidence` is separate from sentiment confidence.
No category means no category_predictions row, and the view's category is NULL.

Only feedback, department, sentiment, confidence, category and category_confidence
are sent. Arbitrary extra CSV columns and employee identifiers are not persisted.
A missing department stays NULL in responses, displayed as "Unknown" in the view.
The CSV `date` is not stored by this schema: dashboard month comes from the batch's
server `uploaded_at`, exactly as in message.txt. Response creation time is also
assigned by the server. Confidence values can be NULL or between 0 and 1.

Each save creates one new integer batch ID and returns its row count. The old
`--table` and UUID `--batch-id` options have been removed.
### Saving flow in storage.py (Week 12 style)

1. `.table("survey_batches").insert(...).execute()` creates the upload with row_count=0.
2. `get_or_create_reference()` finds a department ID, or creates a new department.
3. `.table("responses").insert(...).execute()` stores each message and returns its ID.
4. `.table("sentiment_predictions").insert(...).execute()` links the result to that ID.
5. Optional categories are resolved and saved into category_predictions.
6. `.update({"row_count": ...})` marks the upload complete.

Every database operation is a normal Python Supabase query, like the lab.
Each operation commits separately: this is not a single transaction.
On failure after obtaining a batch ID, Python deletes ONLY that failed batch;
foreign-key CASCADE removes its responses and predictions. Shared department/category
references remain. If cleanup fails, the error identifies the batch for manual checking.
If batch creation succeeds but the ID response is lost, Python cannot safely identify
that upload to delete it. The error asks you to check survey_batches before retrying.
A process crash can also leave a partial upload with row_count=0.

The old save_survey_batch.sql file is no longer needed. An already-installed database
function may remain unused; this code does not call or remove it.

Each successful invocation creates a new batch even for the same file/name.
There are no automatic retries. If a request times out, check survey_batches before
retrying because the database may already have committed the upload.
This simple row-by-row approach suits small demonstrations; large files incur many
network requests. It can be optimized later without changing the model.

## Use from Python

```python
from sentiment.model import load_model, predict_sentiment
from sentiment.storage import init_connection, save_predictions

model = load_model()
result = df.join(predict_sentiment(df["feedback"], model))
supabase = init_connection()
supabase.auth.sign_in_with_password({"email": "you@example.com", "password": "..."})
batch_id, count = save_predictions(
    result, batch_name="Monthly survey", client=supabase
)
```

There is no model fitting or automatic model update when data is uploaded.
Supabase stores results; the Python server runs the model.
