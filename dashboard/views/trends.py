import plotly.express as px
import streamlit as st

from climate import SENTIMENT_COLORS, SENTIMENT_ORDER, filtered_data, require_data, summarize

st.title("Trends Over Time")
df = filtered_data()
require_data(df)

st.subheader("Sentiment share by month")
share = df.groupby("month")["sentiment"].value_counts(normalize=True).mul(100).rename("share").reset_index()
fig = px.area(share, x="month", y="share", color="sentiment", category_orders={"sentiment": SENTIMENT_ORDER},
              color_discrete_map=SENTIMENT_COLORS, labels={"share": "% of responses", "month": ""})
fig.update_layout(yaxis_range=[0, 100], margin=dict(t=10, b=10), legend_title_text="")
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Climate Index by department")
    by_dept = summarize(df, ["month", "department"])
    fig = px.line(by_dept, x="month", y="climate_index", color="department", markers=True,
                  labels={"climate_index": "Climate Index", "month": "", "department": ""})
    fig.update_layout(yaxis_range=[0, 100], margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Climate Index by category")
    by_cat = summarize(df, ["month", "category"])
    fig = px.line(by_cat, x="month", y="climate_index", color="category", markers=True,
                  labels={"climate_index": "Climate Index", "month": "", "category": ""})
    fig.update_layout(yaxis_range=[0, 100], margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

months = sorted(df["month"].unique())
if len(months) > 1:
    st.subheader(f"Biggest changes: {months[0]} → {months[-1]}")
    first = summarize(df[df["month"] == months[0]], "department").set_index("department")["climate_index"]
    last = summarize(df[df["month"] == months[-1]], "department").set_index("department")["climate_index"]
    change = (last - first).dropna().sort_values().rename("change").reset_index()
    fig = px.bar(change, x="change", y="department", orientation="h",
                 color=change["change"] > 0, color_discrete_map={True: "#2E9E6B", False: "#D9534F"},
                 labels={"change": "Change in Climate Index", "department": ""})
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
