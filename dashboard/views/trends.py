import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from climate import (
    HEALTHY_THRESHOLD,
    INDEX_COLORSCALE,
    MIN_RESPONSES,
    SENTIMENT_COLORS,
    STATUS_COLORS,
    WATCH_THRESHOLD,
    compare_windows,
    filtered_data,
    require_data,
    rolling_index,
    split_windows,
    window_label,
)

st.title("Trends Over Time")
df = filtered_data()
require_data(df)
st.caption("Every line here is a 3-month rolling Climate Index: each point pools that month with the two before it. "
           "With 12–70 responses per department per month, raw monthly values jump 20–40 points by chance.")

# --- Biggest changes ------------------------------------------------------------------------------
_, _, cur_months, prev_months = split_windows(df)
if prev_months:
    st.subheader(f"Biggest changes: {window_label(prev_months)} → {window_label(cur_months)}")
    change = compare_windows(df, "department").dropna(subset=["change"]).sort_values("change")
    fig = px.bar(change, x="change", y="department", orientation="h",
                 color=change["change"] > 0,
                 color_discrete_map={True: SENTIMENT_COLORS["Positive"], False: SENTIMENT_COLORS["Negative"]},
                 hover_data={"previous_index": ":.1f", "climate_index": ":.1f", "responses": True},
                 labels={"change": "Change in Climate Index", "department": "",
                         "previous_index": "Before", "climate_index": "Now"})
    fig.add_vline(x=0, line_width=1, line_color="#8A8983")
    fig.update_layout(showlegend=False, bargap=0.35, height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# --- One small chart per department ---------------------------------------------------------------
st.subheader("Climate Index by department")
by_dept = rolling_index(df, "department")
fig = px.line(by_dept, x="month", y="climate_index", facet_col="department", facet_col_wrap=4,
              markers=True, hover_data={"responses": True},
              labels={"climate_index": "Index", "month": ""}, height=480)
fig.update_traces(line_color=SENTIMENT_COLORS["Positive"], line_width=2, marker_size=6)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.add_hrect(y0=0, y1=WATCH_THRESHOLD, fillcolor=STATUS_COLORS["Needs attention"], opacity=0.08, line_width=0,
              row="all", col="all")
fig.add_hrect(y0=WATCH_THRESHOLD, y1=HEALTHY_THRESHOLD, fillcolor=STATUS_COLORS["Watch"], opacity=0.08,
              line_width=0, row="all", col="all")
fig.update_yaxes(range=[20, 80])
fig.update_xaxes(dtick="M3", tickformat="%b")
fig.update_layout(margin=dict(t=30, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("Shaded bands: red = needs attention (< 45), yellow = watch (45–60).")

# --- Topic × month heatmap ------------------------------------------------------------------------
st.subheader("Climate Index by topic")
by_cat = rolling_index(df, "category")
ci = by_cat.pivot(index="category", columns="month", values="climate_index")
n = by_cat.pivot(index="category", columns="month", values="responses").reindex_like(ci)
masked = ci.where(n >= MIN_RESPONSES)
if masked.isna().all().all():
    st.info(f"Not enough responses yet: every cell has fewer than {MIN_RESPONSES}.", icon="ℹ️")
else:
    fig = go.Figure(go.Heatmap(
        z=masked.values, x=list(ci.columns), y=list(ci.index), zmin=20, zmax=80, zmid=50,
        colorscale=INDEX_COLORSCALE, xgap=2, ygap=2,
        text=[[f"{v:.0f}" if v == v else "–" for v in row] for row in masked.values], texttemplate="%{text}",
        customdata=n.values, colorbar=dict(title="Index"),
        hovertemplate="%{y} · %{x}<br>Climate Index %{z:.1f}<br>%{customdata} responses<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=360, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
