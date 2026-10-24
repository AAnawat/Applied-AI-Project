import streamlit as st

from climate import SENTIMENT_ORDER, filtered_data, require_data

REVIEW_THRESHOLD = 0.65

st.title("Feedback Explorer")
df = filtered_data()
require_data(df)

c1, c2, c3 = st.columns([1, 1, 2])
sentiments = c1.multiselect("Sentiment", SENTIMENT_ORDER, placeholder="All")
categories = c2.multiselect("Category", sorted(df["category"].unique()), placeholder="All")
query = c3.text_input("Search text", placeholder="e.g. โอที, เงินเดือน")
review = st.toggle(
    f"Only low-confidence predictions (below {REVIEW_THRESHOLD:.2f})",
    help="The sentiment model was trained on social-media text, so it is least sure about polite, indirect "
         "workplace comments. These are the rows worth a human check.",
)

if review:
    df = df[df["confidence"] < REVIEW_THRESHOLD]
if sentiments:
    df = df[df["sentiment"].isin(sentiments)]
if categories:
    df = df[df["category"].isin(categories)]
if query:
    df = df[df["feedback"].str.contains(query, case=False, regex=False)]

with st.expander("Sort options", expanded=False):
    sort_col, sort_order = st.columns(2)

    sort_options = {
        "Date": "date",
        "Confidence": "confidence",
        "Sentiment": "sentiment",
        "Department": "department",
        "Category": "category",
    }

    # Review mode starts on least-confident first
    selected_label = sort_col.selectbox(
        "Sort by",
        options=list(sort_options.keys()),
        index=1 if review else 0,
    )

    sort_by = sort_options[selected_label]

    order = sort_order.selectbox(
        "Order",
        ["Descending", "Ascending"],
        index=1 if review else 0,
    )

    df = df.sort_values(
        by=sort_by,
        ascending=(order == "Ascending"),
        na_position="last",
        kind="stable",
    )

st.caption(f"{len(df):,} responses")

total = len(df)
negative_count = (df["sentiment"] == "Negative").sum()
positive_count = (df["sentiment"] == "Positive").sum()

m1, m2, m3 = st.columns(3)
m1.metric("Total Responses", f"{total:,}")
m2.metric("Positive", f"{positive_count:,}")
m3.metric("Negative", f"{negative_count:,}")


st.dataframe(
    df[["response_id", "date", "department", "category", "sentiment", "confidence", "feedback"]],
    hide_index=True,
    width="stretch",
    height=520,
    column_config={
        "date": st.column_config.DateColumn("Date"),
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f", color="gray"),
        "feedback": st.column_config.TextColumn("Feedback", width="large"),
    },
)
st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "climatelens_feedback.csv", "text/csv")
