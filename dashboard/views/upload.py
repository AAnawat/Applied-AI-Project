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

df = raw[raw["feedback"].fillna("").astype(str).str.strip() != ""].copy()
skipped = len(raw) - len(df)
df["department"] = df["department"].fillna("Unknown") if "department" in df else "Unknown"
df["date"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.NaT
df["date"] = df["date"].fillna(pd.Timestamp.today().normalize())
df["month"] = df["date"].dt.strftime("%Y-%m")
df.insert(0, "response_id", [f"U{i:05d}" for i in range(1, len(df) + 1)])

with st.spinner("Classifying responses..."):
    df = df.join(classify(df["feedback"]))

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


c1, c2 = st.columns(2)
c1.button("Use this data in the dashboard", type="primary", on_click=use_uploaded, args=(df,))
c2.download_button("Download results", df.to_csv(index=False).encode("utf-8-sig"),
                   "climatelens_results.csv", "text/csv")
