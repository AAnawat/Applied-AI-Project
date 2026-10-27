import streamlit as st

from categorization.core.llm import list_models
from classifier import (
    OVERRIDE_KEY, TopicCategorizationError, check_local_host, check_settings, default_topic_settings,
    override_allowed, topic_override,
)

st.title("Settings")

if not override_allowed():
    st.info("Changing the Local LLM server is turned off. Set `allow_llm_override = true` in Streamlit secrets to enable it.")
    st.stop()

st.subheader("Topic categorization (Local LLM)")
st.caption(
    "Topics are analyzed by an Ollama server on the local network. A server chosen here applies only to "
    "your session; signing out or refreshing the page returns to the default."
)

try:
    default = default_topic_settings()
except TopicCategorizationError as error:
    st.error(str(error))
    st.stop()
override = topic_override()
current = override or default

if st.session_state.pop("llm_saved", False):
    st.success("Saved. The Upload page now analyzes topics with this server.")

with st.container(border=True):
    st.markdown(f"**Currently using:** {'Custom server (this session)' if override else 'Default server'}")
    st.markdown(f"Model: `{current['model']}`  \nServer: `{current['host']}`  \nTimeout: {current['timeout']:g} s")
    if override and st.button("Reset to default"):
        st.session_state.pop(OVERRIDE_KEY, None)
        st.session_state.pop("llm_check", None)
        st.rerun()

host = st.text_input("Ollama server", value=current["host"], placeholder="http://10.x.x.x:11434")

if st.button("Check server"):
    st.session_state.pop("llm_check", None)
    try:
        checked = check_settings(current["model"], host, current["timeout"], where="Settings")
        check_local_host(checked["host"])
        with st.spinner(f"Contacting {checked['host']}..."):
            models = list_models(checked["host"])
    except RuntimeError as error:
        st.error(str(error))
    else:
        st.session_state["llm_check"] = {"host": checked["host"], "models": models}

# Only a server that answered can be saved, and only with a model it has pulled
check = st.session_state.get("llm_check")
if check is None or check["host"] != host.strip().rstrip("/"):
    st.caption("Check the server to list its models.")
    st.stop()
if not check["models"]:
    st.warning("This server is reachable but has no models. Run `ollama pull <model>` on it first.")
    st.stop()

st.success(f"Reachable, {len(check['models'])} model(s) available.")
models = check["models"]
model = st.selectbox("Model", models, index=models.index(current["model"]) if current["model"] in models else 0)
timeout = st.number_input("Timeout per response (seconds)", min_value=10, max_value=600,
                          value=min(max(int(current["timeout"]), 10), 600), step=10)

if st.button("Save", type="primary"):
    st.session_state[OVERRIDE_KEY] = {"host": check["host"], "model": model, "timeout": float(timeout)}
    st.session_state.pop("llm_check", None)
    st.session_state["llm_saved"] = True
    st.rerun()
