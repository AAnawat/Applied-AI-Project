# ClimateLens: developer reference

The Thai quick start is in [../README.md](../README.md). This page is the technical reference.

Streamlit dashboard for Thai employee feedback, sentiment and organizational climate.

## Setup

Run commands from the repository root:

```bash
conda create -n climatelens python=3.11 pip -y
conda activate climatelens
python -m pip install -r requirements.txt
```

### Supabase connection: `.streamlit/secrets.toml`

The dashboard reads its database connection from `.streamlit/secrets.toml`. Without it,
sign-in fails with "Set supabase_url and supabase_key in .streamlit/secrets.toml".

1. Copy the example (it is committed; the real file is ignored by Git):

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

   On Windows Command Prompt: `copy .streamlit\secrets.toml.example .streamlit\secrets.toml`

2. Fill in the two values from the ClimateLens Supabase project. If you are not a member
   of the project, ask whoever owns it for them:

   ```toml
   # Project Settings -> Data API -> Project URL
   supabase_url = "https://your-project.supabase.co"
   # Project Settings -> API Keys -> Publishable key (sb_publishable_...),
   # or the legacy "anon public" key
   supabase_key = "sb_publishable_your-key"
   ```

   Use the **publishable** key only. The Secret / `service_role` key skips Row Level
   Security, so every user would see everyone's data; the app refuses to connect with it.
   Never commit or paste the secret key anywhere.

3. Start the app **from the repository root**. Streamlit looks for `.streamlit/secrets.toml`
   in the folder you run it from, so running it from `dashboard/` finds no secrets:

   ```bash
   python -m streamlit run dashboard/app.py
   ```

   Restart Streamlit after editing the file. In Streamlit hosting, paste the same two
   lines into the app's **Settings -> Secrets** instead.

The Upload & Analyze page and prediction command use the exported sentiment model.
Topic categories in the upload page use the original Typhoon2.5-Qwen3-4B model
through Ollama, zero-shot (no fine-tuned adapter). Start Ollama and pull the model
as described under [Topic categorization](#topic-categorization-ollama) before analyzing.
After uploading a CSV, review the results, enter a survey batch name and click
**Save to Supabase**. The page shows the database batch ID on success and disables
repeat saves of the same file/name within that session. Merely selecting a file or
clicking **Use this data in the dashboard** does not save it to the database.
Users sign in with their Supabase account first (or create one on the **Register** tab);
the dashboard then shows the responses
saved in that account (**Saved**), or an unsaved upload (**Uploaded**). A browser refresh
starts a new Streamlit session, so it signs the user out.

## Structure

- `dashboard/app.py`: entry point, sign-in gate, navigation and global filters (months, departments, data source)
- `dashboard/auth.py`: Supabase Auth sign-in and registration; one Supabase client per browser session
- `dashboard/views/`: Overview, Departments, Trends, Feedback Explorer, Upload & Analyze, Settings
- `dashboard/climate.py`: data access, filters, Climate Index (`50 + 50 × (pos − neg) / n`; ≥60 Healthy, 45–60 Watch, <45 Needs attention)
- `dashboard/classifier.py`: saved sentiment model + zero-shot Ollama topic categorizer; canonical category codes

Dashboard pages show an upload prompt when the selected data source has no responses. If filters exclude all responses, clear or broaden the filters. Uploaded CSVs must contain at least one non-empty `feedback` value.

## Model

Place the exported complete scikit-learn Pipeline (fitted TF-IDF + classifier)
at `models/sentiment.joblib`, or pass `--model /path/to/model.joblib`.
Use the same scikit-learn version as the export environment.
Only load trusted joblib files. This project includes prediction only, with no model fitting.

## Database

The previous single `feedback_predictions` table is no longer used.
The code now follows these tables and views:

| Table/view | Purpose |
| --- | --- |
| `departments` | Unique department names |
| `categories` | Unique category codes |
| `survey_batches` | One upload, server upload time and row count |
| `responses` | Feedback text linked to a batch, optional department and optional survey date |
| `sentiment_predictions` | One sentiment label/probability per response |
| `category_predictions` | Optional category result per response |
| `v_analyzed_responses` | Joined records for the dashboard |
| `v_climate_monthly` | Climate Index grouped by survey month (or upload month when undated), department, category |

Run `supabase/schema.sql` in your Supabase SQL Editor. It creates the tables/views
and backend permissions, including deletion of failed upload batches.
The script can be rerun on matching existing tables to apply permissions. It does not
migrate an incompatible schema or delete existing data.
This is manual setup; Python does not run the SQL automatically.
No additional SQL function or RPC setup is needed.

Then run `supabase/migrations/20261025000000_auth_rls.sql` the same way. It gives
every batch, response and department an `owner_id` (the signed-in user), turns on Row
Level Security so each user reads and writes only their own rows, and makes the views
apply those rules. If the database already has rows, first create the account that
should own them and set `owner_email` at the top of the file.

Then run `supabase/migrations/20261025120000_survey_date.sql`. It stores each
response's CSV `date` in `responses.survey_date` and makes the views count a response
towards that date's month. Rows saved without a date, including everything saved before
this migration, keep counting towards the day they were uploaded. Until it is run, saving
a CSV with a `date` column fails with `PGRST204`.

### Accounts

People register themselves on the dashboard's **Register** tab. For that to work, turn
on **Allow new users to sign up** under **Authentication → Sign In / Providers**. A new
account starts empty: Row Level Security shows it only the batches it uploads itself.

With **Confirm email** on (the default), Supabase emails a link and the user can sign in
only after clicking it. Supabase's built-in mailer sends just a few emails per hour, so
for a class demo either turn **Confirm email** off or set up custom SMTP. Set
**Site URL** under **Authentication → URL Configuration** to the dashboard's address so
the link leads back to it.

Admins can still create users under **Authentication → Users → Add user** (tick
**Auto Confirm User**). To close registration again, turn **Allow new users to sign up**
off; the Register tab then tells people to ask an admin for an account.

### Connection settings

Setting up `.streamlit/secrets.toml` is covered under [Setup](#supabase-connection-streamlitsecretstoml).
The old `.env` configuration is not used.

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
or "Single feedback" for a text message. Batch names do not control the month;
the CSV `date` column does.

To preview without any database connection or write:

```bash
python -m sentiment.predict --text "วันนี้มีประชุม" --dry-run
```

Input CSV requires `feedback`. Optional `department` maps to a department reference.
Optional `date` (`YYYY-MM-DD`) is the survey day the response counts towards.
The prediction command appends sentiment and confidence; input must not already
contain those columns.

Optional `category` must be the category code to store (for example `workload`).
It is preserved if already supplied by a categorizer; no category is invented by
the sentiment model. `category_confidence` is separate from sentiment confidence.
No category means no category_predictions row, and the view's category is NULL.

Only feedback, department, date, sentiment, confidence, category and category_confidence
are sent. Arbitrary extra CSV columns and employee identifiers are not persisted.
A missing department stays NULL in responses, displayed as "Unknown" in the view.
The CSV `date` is stored as `responses.survey_date` and decides the dashboard month;
a row without one falls back to the batch's server `uploaded_at`. Response creation
time is assigned by the server. Confidence values can be NULL or between 0 and 1.

Each save creates one new integer batch ID and returns its row count. The old
`--table` and UUID `--batch-id` options have been removed.

### Saving flow in storage.py

1. `.table("survey_batches").insert(...).execute()` creates the upload with row_count=0.
2. `get_or_create_reference()` finds a department ID, or creates a new department.
3. `.table("responses").insert(...).execute()` stores each message and returns its ID.
4. `.table("sentiment_predictions").insert(...).execute()` links the result to that ID.
5. Optional categories are resolved and saved into category_predictions.
6. `.update({"row_count": ...})` marks the upload complete.

Every database operation is a normal Python Supabase query.
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

## Topic categorization (Ollama)

Single-label: one category code per response, matching `category_predictions`.
Install [Ollama](https://ollama.com/download), start the Ollama application (or
`ollama serve` in a separate terminal), then pull the tested model:

```bash
ollama pull hf.co/typhoon-ai/typhoon2.5-qwen3-4b-gguf:Q4_K_M
```

Start Streamlit from this repository's root. Upload a CSV with `feedback` and,
optionally, `department` / `date`. Do not include existing category or sentiment
result columns: the upload page predicts every row itself. Review the predictions,
then click **Save to Supabase** to persist them through the signed-in user's client.
No SQL changes are required for LLM topics.

Optional settings in `.streamlit/secrets.toml` (also shown in the example file):

```toml
ollama_host = "http://127.0.0.1:11434"
ollama_model = "hf.co/typhoon-ai/typhoon2.5-qwen3-4b-gguf:Q4_K_M"
ollama_timeout = 120
```

The model uses temperature 0, context 2048, thinking disabled and the shared JSON
category schema. A connection error or invalid category stops analysis before saving;
there is no keyword fallback. The most recent successful analysis is kept only in
the browser's Streamlit session, keyed by file contents and Ollama settings, so typing
a batch name or saving does not call the LLM again. Signing out clears the session.
Only feedback text is sent to Ollama, not Supabase credentials or the other CSV fields.
The displayed confidence is the sentiment probability; no topic confidence is invented.
Older keyword category display names are normalized to the same codes when the dashboard
reads them, without rewriting the database.

For a deployed app, `127.0.0.1` means the machine running Streamlit, not the visitor's
computer. The app server must be able to reach its Ollama service; configure
`ollama_host` for a trusted server if the two run on different machines. Do not expose
an unprotected Ollama endpoint publicly.

### Switching the Ollama server from the app

With `allow_llm_override = true` in secrets, the sidebar gets a **Settings** page. Type
another Ollama server's address (for example a team member's PC with a GPU), click
**Check server** to list the models it has pulled, pick one and **Save**. Only your
session changes: other users keep the default, and signing out or refreshing resets it.
The Upload page links to Settings and names the server that analyzed each file.

The Streamlit server makes these requests, so the other machine must accept connections
from it, not from your browser. On that machine, start Ollama with `OLLAMA_HOST=0.0.0.0:11434`
and allow inbound TCP 11434 from the app server only. Settings accepts private and loopback
addresses only (such as `10.x.x.x` or `192.168.x.x`), never internet or link-local ones.

See [categorization/README.md](../categorization/README.md) for the exam/answer-key
evaluation layout. Categorize a CSV, then pass it to the sentiment command, which
keeps the `category` column:

```bash
python -m categorization.cmd.predict --input feedback.csv --output categorized.csv
python -m sentiment.predict --input categorized.csv --dry-run
python -m categorization.cmd.evaluate --exam climatelens_105
```

## Deploy with Docker

`docker-compose.yml` runs the dashboard and Ollama together on one server (tested on an
Ubuntu 24.04 VM with 8 vCPU, 8 GB RAM, no GPU). Ollama is capped at 6 CPUs and 5 GB so
other services keep some headroom, and it is not published on any port: only the app
container can reach it.

1. Copy the repository to the server, plus `.streamlit/secrets.toml`, which Git does not track.
2. In `.streamlit/secrets.toml`, point the app at the Ollama container:

   ```toml
   ollama_host = "http://ollama:11434"
   ```

3. Build and start everything from the repository root. The first start downloads the
   Typhoon model (about 2.5 GB); later starts reuse it:

   ```bash
   docker compose up -d --build
   ```

4. Open `http://<server-ip>:8501`. `docker compose ps` should show `app` as healthy.

All services restart automatically after a reboot. On a CPU without a GPU, topic analysis
takes a few seconds per response (3.6 s/row on the 8-vCPU test VM), and the Upload page
shows per-row progress. Use `docker compose logs -f app` to follow the app's log, and
`docker compose up -d --build app` after pulling new code.

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
