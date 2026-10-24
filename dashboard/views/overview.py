import plotly.express as px
import streamlit as st

from climate import (
    SENTIMENT_COLORS,
    SENTIMENT_ORDER,
    STATUS_COLORS,
    climate_index,
    climate_status,
    filtered_data,
    require_data,
    summarize,
)

st.title("Organizational Climate Overview")
df = filtered_data()
require_data(df)

months = sorted(df["month"].unique())
latest = df[df["month"] == months[-1]]
previous = df[df["month"] == months[-2]] if len(months) > 1 else None

ci_all = climate_index(df["sentiment"])
ci_latest = climate_index(latest["sentiment"])
ci_delta = ci_latest - climate_index(previous["sentiment"]) if previous is not None else None
neg_pct = (df["sentiment"] == "Negative").mean() * 100

dept_latest = summarize(latest, "department")
attention = dept_latest[dept_latest["status"] == "Needs attention"].sort_values("climate_index")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Responses", f"{len(df):,}", help="Survey responses in the selected period")
c2.metric(
    f"Climate Index ({months[-1]})",
    f"{ci_latest:.1f}",
    f"{ci_delta:+.1f} vs prev. month" if ci_delta is not None else None,
    help="50 + 50 × (positive − negative) / responses. 0–100, 50 = balanced.",
)
c3.metric("Negative feedback", f"{neg_pct:.1f}%")
c4.metric("Depts needing attention", len(attention), help=f"Climate Index below threshold in {months[-1]}")

if not attention.empty:
    names = ", ".join(f"**{r.department}** ({r.climate_index:.0f})" for r in attention.itertuples())
    st.error(f"Needs attention in {months[-1]}: {names}", icon="⚠️")

left, right = st.columns([1, 2])
with left:
    st.subheader("Sentiment distribution")
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0).reset_index()
    fig = px.pie(counts, names="sentiment", values="count", hole=0.55, color="sentiment",
                 color_discrete_map=SENTIMENT_COLORS)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Climate Index by category")
    by_cat = summarize(df, "category").sort_values("climate_index")
    fig = px.bar(by_cat, x="climate_index", y="category", orientation="h", color="status",
                 color_discrete_map=STATUS_COLORS, text=by_cat["climate_index"].round(1),
                 hover_data={"responses": True, "negative_pct": ":.1f"},
                 labels={"climate_index": "Climate Index", "category": ""})
    fig.update_layout(xaxis_range=[0, 100], margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

left, right = st.columns([2, 1])
with left:
    st.subheader("Monthly Climate Index")
    trend = summarize(df, "month")
    fig = px.line(trend, x="month", y="climate_index", markers=True,
                  labels={"climate_index": "Climate Index", "month": ""})
    fig.add_hrect(y0=0, y1=45, fillcolor=STATUS_COLORS["Needs attention"], opacity=0.07, line_width=0)
    fig.add_hrect(y0=45, y1=60, fillcolor=STATUS_COLORS["Watch"], opacity=0.07, line_width=0)
    fig.update_layout(yaxis_range=[20, 80], margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Top negative themes")
    negatives = df[df["sentiment"] == "Negative"]
    for category, count in negatives["category"].value_counts().head(4).items():
        example = negatives.loc[negatives["category"] == category, "feedback"].mode().iloc[0]
        with st.container(border=True):
            st.markdown(f"**{category}** · {count} negative responses")
            st.caption(f"“{example}”")

st.subheader("Department snapshot")
dept_all = summarize(df, "department").sort_values("climate_index")
st.dataframe(
    dept_all,
    hide_index=True,
    width="stretch",
    column_config={
        "department": "Department",
        "responses": "Responses",
        "positive_pct": st.column_config.NumberColumn("Positive %", format="%.1f"),
        "neutral_pct": st.column_config.NumberColumn("Neutral %", format="%.1f"),
        "negative_pct": st.column_config.NumberColumn("Negative %", format="%.1f"),
        "climate_index": st.column_config.ProgressColumn("Climate Index", min_value=0, max_value=100, format="%.1f"),
        "status": "Status",
    },
)
