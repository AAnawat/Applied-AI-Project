from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import classify
from sentiment.storage import save_predictions

st.title("Upload & Analyze")
st.markdown(
    "Upload a CSV of survey responses. Required column: `feedback`. "
    "Optional: `department`, `date`."
)
st.caption("Sentiment uses the saved model. Topic categories currently use keyword matching.")
st.info("Upload and review the results, then click Save to Supabase to store them in the database.")

template = pd.DataFrame(
    {
        "feedback": ["หัวหน้ารับฟังความคิดเห็นดี", "งานเยอะเกินไป ต้องทำโอทีทุกวัน", "เงินเดือนต่ำกว่าตลาด"],
        "department": ["IT", "Production", "Sales"],
        "date": ["2026-10-01", "2026-10-03", "2026-10-05"],
    }
)
st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8-sig"),
                   "climatelens_template.csv", "text/csv")

file = st.file_uploader("Survey responses (CSV)", type="csv")
if file is None:
    st.stop()

try:
    raw = pd.read_csv(file, encoding="utf-8-sig")
except (ValueError, UnicodeError, pd.errors.ParserError):
    st.error("Could not read this CSV. Check that it has a header and uses UTF-8 encoding.")
    st.stop()
if "feedback" not in raw.columns:
    st.error(f"Missing `feedback` column. Found: {', '.join(raw.columns)}")
    st.stop()

reserved = {"response_id", "month", "sentiment", "confidence"} & set(raw.columns)
if reserved:
    st.error(f"Please remove result columns before analyzing: {', '.join(sorted(reserved))}")
    st.stop()

df = raw[raw["feedback"].fillna("").astype(str).str.strip() != ""].copy().reset_index(drop=True)
skipped = len(raw) - len(df)
if df.empty:
    st.warning("No non-empty feedback responses were found. Add at least one response and upload the CSV again.")
    st.stop()

df["department"] = df["department"].fillna("Unknown") if "department" in df else "Unknown"
df["date"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.NaT
df["date"] = df["date"].fillna(pd.Timestamp.today().normalize())
df["month"] = df["date"].dt.strftime("%Y-%m")
df.insert(0, "response_id", [f"U{i:05d}" for i in range(1, len(df) + 1)])

try:
    with st.spinner("Classifying responses..."):
        predictions = classify(df["feedback"])
        if "category" in df.columns:
            predictions = predictions.drop(columns="category")
        df = df.join(predictions)
except (ValueError, FileNotFoundError):
    st.error("Could not load or run the sentiment model. Check models/sentiment.joblib and the package versions used to export it.")
    st.stop()

st.success(f"Analyzed {len(df):,} responses." + (f" Skipped {skipped} empty rows." if skipped else ""))
c1, c2, c3 = st.columns(3)
c1.metric("Departments", df["department"].nunique(), help=", ".join(sorted(df["department"].unique())), border=True)
c2.metric("Months", df["month"].nunique(), f"{df['month'].min()} → {df['month'].max()}", delta_color="off", border=True)
c3.metric("Low-confidence rows", int((df["confidence"] < 0.65).sum()), help="Worth a human check", border=True)
st.dataframe(df[["response_id", "date", "department", "category", "sentiment", "confidence", "feedback"]],
             hide_index=True, width="stretch")


def use_uploaded(data: pd.DataFrame) -> None:
    # Runs before the next rerun, i.e. before the sidebar radio is created
    st.session_state["uploaded_df"] = data
    st.session_state["data_source"] = "Uploaded"


st.subheader("Save analyzed responses")
upload_key = sha256(file.getvalue()).hexdigest()
batch_name = st.text_input("Survey batch name", value=Path(file.name).stem)
save_key = (upload_key, batch_name.strip())
saved_uploads = st.session_state.setdefault("saved_uploads", {})
saved = saved_uploads.get(save_key)

if st.button("Save to Supabase", type="primary", disabled=saved is not None or not batch_name.strip()):
    try:
        with st.spinner("Saving responses to Supabase..."):
            batch_id, count = save_predictions(df, batch_name=batch_name)
        saved_uploads[save_key] = (batch_id, count)
        st.rerun()
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
    except Exception:
        st.error("Could not connect to Supabase. Check the server credentials, network and database setup.")

if saved is not None:
    st.success(f"Saved {saved[1]:,} responses to Supabase. Survey batch ID: {saved[0]}")
else:
    st.caption("Results are not saved to the database until you click Save to Supabase.")

c1, c2 = st.columns(2)
c1.button("Use this data in the dashboard", on_click=use_uploaded, args=(df,))
c2.download_button("Download results", df.to_csv(index=False).encode("utf-8-sig"),
                   "climatelens_results.csv", "text/csv")
