import pandas as pd
import streamlit as st

from classifier import classify

st.title("Upload & Analyze")
st.markdown(
    "Upload a CSV of survey responses. Required column: `feedback`. "
    "Optional: `department`, `date`."
)
st.warning("Analysis currently uses a keyword-based placeholder, not the trained models.", icon="🚧")

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

raw = pd.read_csv(file, encoding="utf-8-sig")
if "feedback" not in raw.columns:
    st.error(f"Missing `feedback` column. Found: {', '.join(raw.columns)}")
    st.stop()

df = raw.copy()
df["department"] = df["department"].fillna("Unknown") if "department" in df else "Unknown"
df["date"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.NaT
df["date"] = df["date"].fillna(pd.Timestamp.today().normalize())
df["month"] = df["date"].dt.strftime("%Y-%m")
df.insert(0, "response_id", [f"U{i:05d}" for i in range(1, len(df) + 1)])

with st.spinner("Classifying responses..."):
    df = df.join(classify(df["feedback"]))

st.success(f"Analyzed {len(df):,} responses.")
st.dataframe(df[["response_id", "date", "department", "category", "sentiment", "confidence", "feedback"]],
             hide_index=True, width="stretch")

c1, c2 = st.columns(2)
if c1.button("Use this data in the dashboard", type="primary"):
    st.session_state["uploaded_df"] = df
    st.session_state["data_source"] = "Uploaded"
    st.rerun()
c2.download_button("Download results", df.to_csv(index=False).encode("utf-8-sig"),
                   "climatelens_results.csv", "text/csv")
