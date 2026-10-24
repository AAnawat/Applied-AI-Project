"""ClimateLens dashboard entry point. Run with: streamlit run dashboard/app.py"""

import streamlit as st

from climate import MIN_RESPONSES, get_data

st.set_page_config(page_title="ClimateLens", page_icon="🌡️", layout="wide")

pages = st.navigation(
    [
        st.Page("views/overview.py", title="Overview", icon="📊", default=True),
        st.Page("views/departments.py", title="Departments", icon="🏢"),
        st.Page("views/trends.py", title="Trends", icon="📈"),
        st.Page("views/feedback.py", title="Feedback Explorer", icon="💬"),
        st.Page("views/upload.py", title="Upload & Analyze", icon="📤"),
    ]
)

with st.sidebar:
    st.title("🌡️ ClimateLens")
    st.caption("Organizational Climate Assessment")

    sources = ["Mock data", "Uploaded"] if "uploaded_df" in st.session_state else ["Mock data"]
    source = st.radio("Data source", sources, key="data_source", horizontal=True)

    df = get_data()
    months = sorted(df["month"].unique())
    if len(months) > 1:
        st.select_slider("Months", options=months, value=(months[0], months[-1]), key=f"months_{source}")
        st.session_state["month_range"] = st.session_state[f"months_{source}"]
    else:
        st.session_state["month_range"] = (months[0], months[0]) if months else None

    st.multiselect(
        "Departments",
        sorted(df["department"].unique()),
        key=f"departments_{source}",
        placeholder="All departments",
    )
    st.session_state["departments"] = st.session_state[f"departments_{source}"]

    if source == "Mock data":
        st.info("Showing generated mock data. Real model results will replace it.", icon="ℹ️")

    with st.expander("About the numbers"):
        st.markdown(
            "**Climate Index** = 50 + 50 × (positive − negative) / responses.  \n"
            "🟢 ≥ 60 Healthy · 🟡 45–60 Watch · 🔴 < 45 Needs attention · "
            f"⚪ fewer than {MIN_RESPONSES} responses (no status).\n\n"
            "**Sentiment model:** TF-IDF + Logistic Regression, macro-F1 0.67 on Wisesight (social-media text). "
            "Expect lower accuracy on polite, indirect workplace comments.\n\n"
            "**Topics:** LLM categorizer (in progress)."
        )

pages.run()
