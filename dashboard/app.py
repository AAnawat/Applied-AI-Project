"""ClimateLens dashboard entry point. Run with: streamlit run dashboard/app.py"""

import streamlit as st

from climate import get_data

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

pages.run()
