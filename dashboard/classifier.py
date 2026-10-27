"""Saved sentiment model and zero-shot Local LLM topics for the Upload page."""

import ipaddress
import math
import socket
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from categorization.core.llm import DEFAULT_HOST, DEFAULT_MODEL, categorize
from sentiment.model import load_model, predict_sentiment

# Session key for the Ollama server chosen on the Settings page; it lasts until sign-out or refresh
OVERRIDE_KEY = "llm_override"


class TopicCategorizationError(RuntimeError):
    """Topic analysis failed; no keyword fallback or partial save is allowed."""


def override_allowed():
    """The Settings page is opt-in: the server makes the requests, so only trusted deployments enable it."""
    try:
        return st.secrets.get("allow_llm_override", False) is True
    except FileNotFoundError:
        return False


def check_settings(model, host, timeout, where="Streamlit secrets"):
    if not isinstance(model, str) or not model.strip():
        raise TopicCategorizationError(f"Set a non-empty ollama_model in {where}.")

    if not isinstance(host, str):
        raise TopicCategorizationError("Set ollama_host to the Ollama server URL.")

    try:
        url = urlsplit(host.strip())
        url.port  # raises ValueError for a port such as :abc
        timeout = float(timeout)
    except (ValueError, TypeError):
        raise TopicCategorizationError(f"Check ollama_host and ollama_timeout in {where}.") from None

    if url.scheme not in {"http", "https"} or not url.hostname:
        raise TopicCategorizationError("ollama_host must be an http:// or https:// server URL.")

    if not math.isfinite(timeout) or timeout <= 0:
        raise TopicCategorizationError("ollama_timeout must be a positive number of seconds.")

    return {"model": model.strip(), "host": host.strip().rstrip("/"), "timeout": timeout}


def check_local_host(host):
    """Reject servers outside the local network, so Settings cannot send requests to the internet."""
    name = urlsplit(host).hostname
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(name, None)}
    except (socket.gaierror, UnicodeError):
        raise TopicCategorizationError(f"Cannot find the server {name}. Check the address.") from None

    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        # Link-local covers cloud metadata endpoints such as 169.254.169.254
        if not (ip.is_private or ip.is_loopback) or ip.is_link_local:
            raise TopicCategorizationError(
                f"{name} is not on the local network. Only private addresses such as 10.x.x.x "
                "or 192.168.x.x can be used."
            )


def default_topic_settings():
    try:
        model = st.secrets.get("ollama_model", DEFAULT_MODEL)
        host = st.secrets.get("ollama_host", DEFAULT_HOST)
        timeout = st.secrets.get("ollama_timeout", 120)
    except FileNotFoundError:
        model, host, timeout = DEFAULT_MODEL, DEFAULT_HOST, 120

    return check_settings(model, host, timeout)


def topic_override():
    """The Settings-page server for this session, or None when the default applies."""
    override = st.session_state.get(OVERRIDE_KEY)
    return override if override and override_allowed() else None


def topic_settings():
    override = topic_override()
    if override is None:
        return default_topic_settings()

    settings = check_settings(override["model"], override["host"], override["timeout"], where="Settings")
    check_local_host(settings["host"])

    return settings


@st.cache_resource(show_spinner="Loading sentiment model...")
def cached_sentiment_model():
    return load_model()


def classify(texts: pd.Series, *, model=DEFAULT_MODEL, host=DEFAULT_HOST, timeout=120, progress=None) -> pd.DataFrame:
    result = predict_sentiment(texts, cached_sentiment_model())

    try:
        topics = categorize(texts, model=model, host=host, timeout=timeout, few_shot=False, progress=progress)
    except (RuntimeError, ValueError, OSError) as error:
        raise TopicCategorizationError(
            f"Could not analyze topics with Local LLM ({model} at {host}). "
            "Start Ollama, pull this model, and check ollama_host/ollama_model "
            "in Streamlit secrets or on the Settings page. No results have been saved."
        ) from error

    if not topics["category_valid"].all():
        invalid = int((~topics["category_valid"]).sum())
        raise TopicCategorizationError(
            f"Local LLM returned {invalid} invalid topic answer(s). "
            "Retry the analysis before saving; no results have been saved."
        )

    result["category"] = topics["category"]

    return result
