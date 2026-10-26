"""Saved sentiment model and zero-shot Local LLM topics for the Upload page."""

import math
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from categorization.core.llm import DEFAULT_HOST, DEFAULT_MODEL, categorize
from sentiment.model import load_model, predict_sentiment


class TopicCategorizationError(RuntimeError):
    """Topic analysis failed; no keyword fallback or partial save is allowed."""


def topic_settings():
    try:
        model = st.secrets.get("ollama_model", DEFAULT_MODEL)
        host = st.secrets.get("ollama_host", DEFAULT_HOST)
        timeout = st.secrets.get("ollama_timeout", 120)
    except FileNotFoundError:
        model, host, timeout = DEFAULT_MODEL, DEFAULT_HOST, 120

    if not isinstance(model, str) or not model.strip():
        raise TopicCategorizationError("Set a non-empty ollama_model in Streamlit secrets.")

    if not isinstance(host, str):
        raise TopicCategorizationError("Set ollama_host to the Ollama server URL.")

    try:
        url = urlsplit(host.strip())
        timeout = float(timeout)
    except (ValueError, TypeError):
        raise TopicCategorizationError("Check ollama_host and ollama_timeout in Streamlit secrets.") from None

    if url.scheme not in {"http", "https"} or not url.netloc:
        raise TopicCategorizationError("ollama_host must be an http:// or https:// server URL.")

    if not math.isfinite(timeout) or timeout <= 0:
        raise TopicCategorizationError("ollama_timeout must be a positive number of seconds.")

    return {"model": model.strip(), "host": host.strip().rstrip("/"), "timeout": timeout}


@st.cache_resource(show_spinner="Loading sentiment model...")
def cached_sentiment_model():
    return load_model()


def classify(texts: pd.Series, *, model=DEFAULT_MODEL, host=DEFAULT_HOST, timeout=120, progress=None) -> pd.DataFrame:
    result = predict_sentiment(texts, cached_sentiment_model())

    try:
        topics = categorize(texts, model=model, host=host, timeout=timeout, few_shot=False, progress=progress)
    except (RuntimeError, ValueError, OSError) as error:
        raise TopicCategorizationError(
            f"Could not analyze topics with Local LLM ({model}). "
            "Start Ollama, pull this model, and check ollama_host/ollama_model "
            "in Streamlit secrets. No results have been saved."
        ) from error

    if not topics["category_valid"].all():
        invalid = int((~topics["category_valid"]).sum())
        raise TopicCategorizationError(
            f"Local LLM returned {invalid} invalid topic answer(s). "
            "Retry the analysis before saving; no results have been saved."
        )

    result["category"] = topics["category"]

    return result
