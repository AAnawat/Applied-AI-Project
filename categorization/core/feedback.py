"""Input checks for feedback text before categorization."""

import pandas as pd


def clean_feedback(texts):
    """Return stripped feedback as a Series (index kept); reject missing or blank text."""
    series = texts.copy() if isinstance(texts, pd.Series) else pd.Series(texts, dtype=object)
    if series.isna().any():
        raise ValueError("Feedback contains missing values; remove them before categorization.")
    cleaned = series.astype(str).str.strip()
    if cleaned.eq("").any():
        raise ValueError("Feedback contains blank text; remove it before categorization.")
    return cleaned
