"""Shared data access, filters and Climate Index calculations for all pages."""

import pandas as pd
import streamlit as st

from mock_data import generate_mock_data

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {"Positive": "#2E9E6B", "Neutral": "#9AA0A6", "Negative": "#D9534F"}
STATUS_COLORS = {"Healthy": "#2E9E6B", "Watch": "#E0A030", "Needs attention": "#D9534F"}

# Climate Index = 50 + 50 * (positive - negative) / responses, so 0..100 with 50 = balanced
HEALTHY_THRESHOLD = 60
WATCH_THRESHOLD = 45


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


def climate_status(ci: float) -> str:
    if ci >= HEALTHY_THRESHOLD:
        return "Healthy"
    if ci >= WATCH_THRESHOLD:
        return "Watch"
    return "Needs attention"


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
    out["status"] = out["climate_index"].map(climate_status)
    return out


def require_data(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No responses match the current filters.")
        st.stop()
