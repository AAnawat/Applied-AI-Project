import streamlit as st

from climate import SENTIMENT_ORDER, filtered_data, require_data

st.title("Feedback Explorer")
df = filtered_data()
require_data(df)

c1, c2, c3 = st.columns([1, 1, 2])
sentiments = c1.multiselect("Sentiment", SENTIMENT_ORDER, placeholder="All")
categories = c2.multiselect("Category", sorted(df["category"].unique()), placeholder="All")
query = c3.text_input("Search text", placeholder="e.g. โอที, เงินเดือน")

if sentiments:
    df = df[df["sentiment"].isin(sentiments)]
if categories:
    df = df[df["category"].isin(categories)]
if query:
    df = df[df["feedback"].str.contains(query, case=False, regex=False)]

st.caption(f"{len(df):,} responses")
st.dataframe(
    df[["response_id", "date", "department", "category", "sentiment", "confidence", "feedback"]],
    hide_index=True,
    width="stretch",
    height=520,
    column_config={
        "date": st.column_config.DateColumn("Date"),
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
        "feedback": st.column_config.TextColumn("Feedback", width="large"),
    },
)
st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "climatelens_feedback.csv", "text/csv")
