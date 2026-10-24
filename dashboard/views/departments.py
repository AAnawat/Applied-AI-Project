import plotly.express as px
import streamlit as st

from climate import SENTIMENT_COLORS, STATUS_COLORS, filtered_data, require_data, summarize

st.title("Department Climate")
df = filtered_data()
require_data(df)

st.subheader("Climate Index: department × category")
grid = summarize(df, ["department", "category"]).pivot(index="department", columns="category", values="climate_index")
fig = px.imshow(grid.round(0), text_auto=True, aspect="auto", zmin=20, zmax=80,
                color_continuous_scale=["#D9534F", "#F2D27A", "#2E9E6B"],
                labels={"color": "Climate Index", "x": "", "y": ""})
fig.update_layout(margin=dict(t=10, b=10))
st.plotly_chart(fig, width="stretch")

st.subheader("Cross-department comparison")
by_dept = summarize(df, "department").sort_values("climate_index", ascending=False)
fig = px.bar(by_dept, x="department", y=["positive_pct", "neutral_pct", "negative_pct"],
             labels={"value": "% of responses", "department": "", "variable": ""},
             color_discrete_sequence=[SENTIMENT_COLORS[s] for s in ["Positive", "Neutral", "Negative"]])
fig.for_each_trace(lambda t: t.update(name=t.name.replace("_pct", "").title()))
fig.update_layout(barmode="stack", margin=dict(t=10, b=10))
st.plotly_chart(fig, width="stretch")

st.divider()
dept = st.selectbox("Drill into a department", by_dept.sort_values("climate_index")["department"])
d = df[df["department"] == dept]
row = by_dept.set_index("department").loc[dept]

c1, c2, c3 = st.columns(3)
c1.metric("Climate Index", f"{row.climate_index:.1f}")
c2.metric("Status", row.status)
c3.metric("Responses", f"{int(row.responses):,}")

left, right = st.columns(2)
with left:
    st.markdown("**By category**")
    cat = summarize(d, "category").sort_values("climate_index")
    fig = px.bar(cat, x="climate_index", y="category", orientation="h", color="status",
                 color_discrete_map=STATUS_COLORS, labels={"climate_index": "Climate Index", "category": ""})
    fig.update_layout(xaxis_range=[0, 100], margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
with right:
    st.markdown("**Monthly trend**")
    trend = summarize(d, "month")
    fig = px.line(trend, x="month", y="climate_index", markers=True,
                  labels={"climate_index": "Climate Index", "month": ""})
    fig.update_layout(yaxis_range=[0, 100], margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

st.markdown("**Recurring negative feedback**")
recurring = (
    d[d["sentiment"] == "Negative"]
    .groupby(["category", "feedback"]).size().reset_index(name="mentions")
    .sort_values("mentions", ascending=False).head(8)
)
st.dataframe(recurring, hide_index=True, width="stretch")
