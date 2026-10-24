"""Shared data access, filters and Climate Index calculations for all pages."""

import pandas as pd
import streamlit as st

from mock_data import generate_mock_data

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
# Diverging pair (blue = good, red = bad) with a gray midpoint, checked for colour-blind separation
SENTIMENT_COLORS = {"Positive": "#2A78D6", "Neutral": "#B5B4AD", "Negative": "#E34948"}
STATUS_COLORS = {"Healthy": "#0CA30C", "Watch": "#FAB219", "Needs attention": "#D03B3B", "Not enough data": "#8A8983"}
STATUS_ICONS = {"Healthy": "🟢", "Watch": "🟡", "Needs attention": "🔴", "Not enough data": "⚪"}
# Heatmaps: red (low) -> gray (50 = balanced) -> blue (high)
INDEX_COLORSCALE = [[0, "#E34948"], [0.5, "#D9D8D2"], [1, "#2A78D6"]]
# Table progress bars show magnitude only (one colour for every row); the Status column carries good/bad
INDEX_BAR_COLOR = "#2A78D6"

# Climate Index = 50 + 50 * (positive - negative) / responses, so 0..100 with 50 = balanced
HEALTHY_THRESHOLD = 60
WATCH_THRESHOLD = 45
# Groups smaller than this get no status: the index is too noisy, and tiny groups risk identifying people
MIN_RESPONSES = 10
# Comparisons use 3-month windows; single months swing by 20-40 points on mock-sized departments
WINDOW_MONTHS = 3


@st.cache_data
def load_mock_data() -> pd.DataFrame:
    return generate_mock_data()


def get_data() -> pd.DataFrame:
    """The dataset currently selected in the sidebar (mock or uploaded)."""
    if st.session_state.get("data_source") == "Uploaded" and "uploaded_df" in st.session_state:
        return st.session_state["uploaded_df"]
    return load_mock_data()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    months = st.session_state.get("month_range")
    if months:
        df = df[(df["month"] >= months[0]) & (df["month"] <= months[1])]
    departments = st.session_state.get("departments")
    if departments:
        df = df[df["department"].isin(departments)]
    return df


def filtered_data() -> pd.DataFrame:
    return apply_filters(get_data())


def climate_index(sentiments: pd.Series) -> float:
    if len(sentiments) == 0:
        return float("nan")
    pos = (sentiments == "Positive").sum()
    neg = (sentiments == "Negative").sum()
    return 50 + 50 * (pos - neg) / len(sentiments)


def climate_status(ci: float, responses: int | None = None) -> str:
    if (responses is not None and responses < MIN_RESPONSES) or pd.isna(ci):
        return "Not enough data"
    if ci >= HEALTHY_THRESHOLD:
        return "Healthy"
    if ci >= WATCH_THRESHOLD:
        return "Watch"
    return "Needs attention"


def status_label(status: str) -> str:
    return f"{STATUS_ICONS[status]} {status}"


def summarize(df: pd.DataFrame, by) -> pd.DataFrame:
    """Responses, sentiment shares (%) and Climate Index per group."""
    g = df.groupby(by)["sentiment"]
    out = g.agg(
        responses="size",
        positive_pct=lambda s: (s == "Positive").mean() * 100,
        neutral_pct=lambda s: (s == "Neutral").mean() * 100,
        negative_pct=lambda s: (s == "Negative").mean() * 100,
        climate_index=climate_index,
    ).reset_index()
    out["status"] = [climate_status(ci, n) for ci, n in zip(out["climate_index"], out["responses"])]
    return out


def split_windows(df: pd.DataFrame, size: int = WINDOW_MONTHS):
    """Split into (current, previous, current_months, previous_months): the last `size` months and the `size` before."""
    months = sorted(df["month"].unique())
    current_months = months[-size:]
    previous_months = months[-2 * size:-size] if len(months) > size else []
    return (
        df[df["month"].isin(current_months)],
        df[df["month"].isin(previous_months)],
        current_months,
        previous_months,
    )


def window_label(months: list[str]) -> str:
    if not months:
        return "n/a"
    first, last = pd.Period(months[0]), pd.Period(months[-1])
    if first == last:
        return first.strftime("%b %Y")
    start = first.strftime("%b") if first.year == last.year else first.strftime("%b %Y")
    return f"{start} – {last.strftime('%b %Y')}"


def compare_windows(df: pd.DataFrame, by) -> pd.DataFrame:
    """Per-group Climate Index for the current window, previous window and the change between them."""
    current, previous, _, _ = split_windows(df)
    cur = summarize(current, by)
    prev = summarize(previous, by)[_as_list(by) + ["climate_index", "responses"]] if not previous.empty else None
    if prev is None:
        cur["previous_index"] = float("nan")
        cur["previous_responses"] = 0
    else:
        prev = prev.rename(columns={"climate_index": "previous_index", "responses": "previous_responses"})
        cur = cur.merge(prev, on=_as_list(by), how="left")
    enough = (cur["responses"] >= MIN_RESPONSES) & (cur["previous_responses"].fillna(0) >= MIN_RESPONSES)
    cur["change"] = (cur["climate_index"] - cur["previous_index"]).where(enough)
    return cur


def rolling_index(df: pd.DataFrame, by=None, size: int = WINDOW_MONTHS) -> pd.DataFrame:
    """Climate Index per month, pooled over the trailing `size` months (not an average of monthly averages)."""
    months = sorted(df["month"].unique())
    frames = []
    for i, month in enumerate(months):
        window = df[df["month"].isin(months[max(0, i - size + 1):i + 1])]
        s = summarize(window, by) if by else summarize(window.assign(_all="All"), "_all").drop(columns="_all")
        frames.append(s.assign(month=month))
    return pd.concat(frames, ignore_index=True)


def _as_list(by) -> list:
    return list(by) if isinstance(by, (list, tuple)) else [by]


def require_data(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No responses match the current filters.")
        st.stop()
