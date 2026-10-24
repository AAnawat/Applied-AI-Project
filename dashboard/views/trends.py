import plotly.express as px
import streamlit as st

from climate import SENTIMENT_COLORS, SENTIMENT_ORDER, filtered_data, require_data, summarize

st.title("Trends Over Time")
df = filtered_data()
require_data(df)

st.subheader("Sentiment share by month")
share = (
    df.groupby("month")["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("share")
    .reset_index()
)

fig = px.area(share, x="month", y="share", color="sentiment", category_orders={"sentiment": SENTIMENT_ORDER},
              color_discrete_map=SENTIMENT_COLORS, labels={"share": "% of responses", "month": ""})
fig.update_layout(yaxis_range=[0, 100], margin=dict(t=10, b=10), legend_title_text="")

st.plotly_chart(fig, width="stretch")

if not share.empty:
    latest_month = share["month"].max()
    latest = share[share["month"] == latest_month]

    sentiment_pct = latest.set_index("sentiment")["share"]
    positive = sentiment_pct.get("Positive", 0)
    negative = sentiment_pct.get("Negative", 0)
    neutral = sentiment_pct.get("Neutral", 0)

    dominant = latest.loc[latest["share"].idxmax()]

    st.info(
        f"**Sentiment insight — {latest_month}:** "
        f"{dominant['sentiment']} is the most common sentiment "
        f"({dominant['share']:.1f}%). "
        f"Positive: {positive:.1f}%, "
        f"Neutral: {neutral:.1f}%, "
        f"Negative: {negative:.1f}%."
    )


st.subheader("Climate Index by department")
by_dept = summarize(df, ["month", "department"])
fig = px.line(by_dept, x="month", y="climate_index", color="department", markers=True,
                labels={"climate_index": "Climate Index", "month": "", "department": ""})
fig.update_layout(yaxis_range=[0, 100], margin=dict(t=10, b=10))
st.plotly_chart(fig, width="stretch")

if not by_dept.empty:
    latest_month = by_dept["month"].max()
    latest = by_dept[by_dept["month"] == latest_month].dropna(
        subset=["climate_index"]
    )

    if not latest.empty:
        best = latest.loc[latest["climate_index"].idxmax()]
        worst = latest.loc[latest["climate_index"].idxmin()]

        st.info(
            f"**Department insight — {latest_month}:** "
            f"{best['department']} has the highest Climate Index "
            f"({best['climate_index']:.1f}), while "
            f"{worst['department']} has the lowest "
            f"({worst['climate_index']:.1f})."
        )

st.subheader("Climate Index by category")
by_cat = summarize(df, ["month", "category"])
fig = px.line(by_cat, x="month", y="climate_index", color="category", markers=True,
                labels={"climate_index": "Climate Index", "month": "", "category": ""})
fig.update_layout(yaxis_range=[0, 100], margin=dict(t=10, b=10))
st.plotly_chart(fig, width="stretch")


if not by_cat.empty:
    latest_month = by_cat["month"].max()
    latest = by_cat[by_cat["month"] == latest_month].dropna(
        subset=["climate_index"]
    )

    if not latest.empty:
        best = latest.loc[latest["climate_index"].idxmax()]
        worst = latest.loc[latest["climate_index"].idxmin()]

        st.info(
            f"**Category insight — {latest_month}:** "
            f"{best['category']} has the highest Climate Index "
            f"({best['climate_index']:.1f}), while "
            f"{worst['category']} has the lowest "
            f"({worst['climate_index']:.1f}). "
            f"Consider reviewing the lowest-scoring category."
        )


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

if not change.empty:
    biggest_drop = change.iloc[0]
    biggest_gain = change.iloc[-1]

    st.info(
        f"**Change insight ({months[0]} → {months[-1]}):** "
        f"{biggest_gain['department']} had the largest increase "
        f"(+{biggest_gain['change']:.1f} points), while "
        f"{biggest_drop['department']} had the largest decrease "
        f"({biggest_drop['change']:+.1f} points)."
    )

