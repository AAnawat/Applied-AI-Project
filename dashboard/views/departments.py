import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from climate import (
    HEALTHY_THRESHOLD,
    SENTIMENT_COLORS,
    SENTIMENT_ORDER,
    STATUS_COLORS,
    WATCH_THRESHOLD,
    compare_windows,
    filtered_data,
    require_data,
    rolling_index,
    status_label,
    summarize,
)

st.title("Department Climate")
df = filtered_data()
require_data(df)

# --- Diverging sentiment bar: negative to the left of 0, positive to the right ------------------
st.subheader("Sentiment by department")
st.caption("Negative grows left, positive grows right, neutral is split across the centre line. Sorted by Climate Index.")
by_dept = summarize(df, "department").sort_values("climate_index")
fig = go.Figure()
for name, values, show in [
    ("Neutral", -by_dept["neutral_pct"] / 2, True),
    ("Negative", -by_dept["negative_pct"], True),
    ("Neutral", by_dept["neutral_pct"] / 2, False),
    ("Positive", by_dept["positive_pct"], True),
]:
    actual = values.abs() * (2 if name == "Neutral" else 1)
    fig.add_bar(y=by_dept["department"], x=values, name=name, orientation="h", marker_color=SENTIMENT_COLORS[name],
                showlegend=show, legendgroup=name, legendrank=SENTIMENT_ORDER[::-1].index(name), customdata=actual,
                hovertemplate="%{y}<br>" + name + " %{customdata:.0f}%<extra></extra>")
fig.update_layout(barmode="relative", bargap=0.35, height=360, margin=dict(t=10, b=10),
                  legend=dict(orientation="h", y=1.08, traceorder="normal"),
                  xaxis=dict(title="% of responses", tickvals=[-75, -50, -25, 0, 25, 50, 75],
                             ticktext=["75", "50", "25", "0", "25", "50", "75"]))
fig.add_vline(x=0, line_width=1, line_color="#8A8983")
st.plotly_chart(fig, width="stretch")

# --- Drill-down -----------------------------------------------------------------------------------
st.divider()
dept = st.selectbox("Drill into a department", by_dept["department"])
d = df[df["department"] == dept]
row = compare_windows(df, "department").set_index("department").loc[dept]

c1, c2, c3 = st.columns([1, 1.5, 0.8])
c1.metric("Climate Index (last 3 months)", f"{row.climate_index:.1f}",
          f"{row.change:+.1f} vs prev. 3 months" if row.change == row.change else None, border=True)
c2.metric("Status", status_label(row.status), border=True)
c3.metric("Responses (last 3 months)", f"{int(row.responses):,}", help=f"{len(d):,} in the selected period", border=True)

left, right = st.columns(2)
with left:
    st.markdown("**By topic** (selected period)")
    cat = summarize(d, "category").sort_values("climate_index")
    fig = px.bar(cat, x="climate_index", y="category", orientation="h", color="status",
                 color_discrete_map=STATUS_COLORS, hover_data={"responses": True, "status": False},
                 labels={"climate_index": "Climate Index", "category": ""})
    fig.add_vline(x=50, line_width=1, line_color="#8A8983")
    fig.update_layout(xaxis_range=[0, 100], bargap=0.35, margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
with right:
    st.markdown("**Climate Index over time** (3-month rolling)")
    trend = rolling_index(d)
    fig = px.line(trend, x="month", y="climate_index", markers=True,
                  hover_data={"responses": True}, labels={"climate_index": "Climate Index", "month": ""})
    fig.add_hrect(y0=0, y1=WATCH_THRESHOLD, fillcolor=STATUS_COLORS["Needs attention"], opacity=0.08, line_width=0)
    fig.add_hrect(y0=WATCH_THRESHOLD, y1=HEALTHY_THRESHOLD, fillcolor=STATUS_COLORS["Watch"], opacity=0.08, line_width=0)
    fig.update_traces(line_color=SENTIMENT_COLORS["Positive"], line_width=2, marker_size=8)
    fig.update_xaxes(dtick="M1" if len(trend) <= 6 else "M3", tickformat="%b %Y")
    fig.update_layout(yaxis_range=[20, 80], margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

st.markdown("**What people say**")
st.caption("The comments the model is most confident about, per topic. Real data won't repeat word for word, "
           "so we show examples instead of counting identical sentences.")
neg_tab, pos_tab = st.tabs(["Negative", "Positive"])
for tab, sentiment in [(neg_tab, "Negative"), (pos_tab, "Positive")]:
    examples = (
        d[d["sentiment"] == sentiment]
        .sort_values("confidence", ascending=False)
        .drop_duplicates("feedback")
        .groupby("category").head(2)
        .sort_values("category")
    )
    with tab:
        st.dataframe(examples[["category", "feedback", "confidence", "date"]], hide_index=True, width="stretch",
                     column_config={
                         "category": "Topic",
                         "feedback": st.column_config.TextColumn("Comment", width="large"),
                         "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f", color="gray"),
                         "date": st.column_config.DateColumn("Date"),
                     })
