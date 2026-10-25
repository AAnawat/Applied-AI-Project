"""Supabase Auth login for the dashboard.

Each browser session gets its own Supabase client. After sign-in it sends that user's
token with every query, and Row Level Security in the database returns only the
user's own rows. Never put this client in st.cache_resource: that cache is shared by
every visitor, so they would all become whoever signed in last.
"""

import streamlit as st
from supabase_auth.errors import AuthError

from sentiment.storage import init_connection


def client():
    """This browser session's Supabase client."""
    if "supabase" not in st.session_state:
        st.session_state["supabase"] = init_connection()
    return st.session_state["supabase"]


def current_user() -> dict | None:
    return st.session_state.get("user")


def login_form() -> None:
    """Show the sign-in screen and stop the page until the user is signed in."""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("ClimateLens")
        st.caption("Organizational Climate Assessment. Sign in to see your survey results.")
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")

        if submitted:
            try:
                session = client().auth.sign_in_with_password({"email": email.strip(), "password": password})
            except ValueError as error:
                st.error(str(error))
            except AuthError:
                st.error("Wrong email or password.")
            except Exception:
                st.error("Could not reach Supabase. Check the network and supabase_url.")
            else:
                st.session_state["user"] = {"id": session.user.id, "email": session.user.email}
                st.rerun()
    st.stop()


def logout() -> None:
    try:
        client().auth.sign_out()
    except Exception:
        pass  # The local session is dropped below either way
    st.session_state.clear()
