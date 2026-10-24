import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from climate import (
    INDEX_BAR_COLOR,
    INDEX_COLORSCALE,
    MIN_RESPONSES,
    climate_index,
    compare_windows,
    filtered_data,
    require_data,
    split_windows,
    status_label,
    summarize,
    window_label,
)

st.title("Organizational Climate Overview")
df = filtered_data()
require_data(df)

current, previous, cur_months, prev_months = split_windows(df)


def fmt_change(v: float) -> str:
    return "–" if v != v else f"{v:+.0f}"


st.caption(
    f"Comparing the latest 3 months (**{window_label(cur_months)}**) with the 3 months before "
    f"(**{window_label(prev_months)}**). Single months are too noisy to compare."
)

# --- Headline numbers: all four use the same window --------------------------------------------
ci_now = climate_index(current["sentiment"])
ci_prev = climate_index(previous["sentiment"]) if not previous.empty else None
neg_now = (current["sentiment"] == "Negative").mean() * 100
neg_prev = (previous["sentiment"] == "Negative").mean() * 100 if not previous.empty else None

cells = compare_windows(df, ["department", "category"])
hotspots = cells[cells["status"] == "Needs attention"].sort_values("climate_index")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Climate Index", f"{ci_now:.1f}", f"{ci_now - ci_prev:+.1f}" if ci_prev is not None else None,
          help="50 + 50 × (positive − negative) / responses. 0–100, 50 = balanced.", border=True)
c2.metric("Negative feedback", f"{neg_now:.1f}%", f"{neg_now - neg_prev:+.1f} pts" if neg_prev is not None else None,
          delta_color="inverse", border=True)
c3.metric("Responses", f"{len(current):,}", help="Survey responses in the latest 3 months", border=True)
c4.metric("Hotspots", len(hotspots), help="Department × topic pairs with a Climate Index below 45", border=True)

# --- Where to look first ------------------------------------------------------------------------
st.subheader("Where to look first")
if hotspots.empty:
    st.success("No department × topic pair is below the attention threshold.", icon="✅")
else:
    negatives = current[current["sentiment"] == "Negative"].sort_values("confidence", ascending=False)

    def example(row) -> str:
        match = negatives[(negatives["department"] == row.department) & (negatives["category"] == row.category)]
        return match["feedback"].iloc[0] if not match.empty else ""

    top = hotspots.head(5).copy()
    top["example"] = [example(r) for r in top.itertuples()]
    top["change"] = top["change"].map(fmt_change)
    st.dataframe(
        top[["department", "category", "climate_index", "change", "responses", "example"]],
        hide_index=True,
        width="stretch",
        column_config={
            "department": "Department",
            "category": "Topic",
            "climate_index": st.column_config.ProgressColumn("Climate Index", min_value=0, max_value=100, format="%.0f", color=INDEX_BAR_COLOR),
            "change": "vs prev. 3 months",
            "responses": "Responses",
            "example": st.column_config.TextColumn("Typical negative comment", width="large"),
        },
    )

# --- Department × topic heatmap -----------------------------------------------------------------
months = sorted(df["month"].unique())
st.subheader("Climate Index by department and topic")
st.caption(f"Whole selected period (**{window_label([months[0], months[-1]])}**): three months split across seven "
           "topics leaves too few responses per cell.")
grid = summarize(df, ["department", "category"])
by_cat = summarize(df, "category").assign(department="All departments")
grid = pd.concat([grid, by_cat], ignore_index=True)
ci = grid.pivot(index="department", columns="category", values="climate_index")
n = grid.pivot(index="department", columns="category", values="responses").reindex_like(ci)
order = [d for d in ci.index if d != "All departments"] + ["All departments"]
ci, n = ci.loc[order], n.loc[order]
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
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
st.caption(f"Red = more negative than positive, blue = more positive. “–” = fewer than {MIN_RESPONSES} responses.")

# --- Department table ---------------------------------------------------------------------------
st.subheader("Departments")
depts = compare_windows(df, "department").sort_values("climate_index")
depts["status"] = depts["status"].map(status_label)
depts["change"] = depts["change"].map(fmt_change)
st.dataframe(
    depts[["department", "status", "climate_index", "change", "responses", "positive_pct", "negative_pct"]],
    hide_index=True,
    width="stretch",
    column_config={
        "department": "Department",
        "status": "Status",
        "climate_index": st.column_config.ProgressColumn("Climate Index", min_value=0, max_value=100, format="%.1f", color=INDEX_BAR_COLOR),
        "change": "vs prev. 3 months",
        "responses": "Responses",
        "positive_pct": st.column_config.NumberColumn("Positive %", format="%.0f"),
        "negative_pct": st.column_config.NumberColumn("Negative %", format="%.0f"),
    },
)
