"""Supabase Auth sign-in and registration for the dashboard.

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


def _start_session(user) -> None:
    st.session_state["user"] = {"id": user.id, "email": user.email}
    st.rerun()


def login_form() -> None:
    """Show the sign-in and register screen and stop the page until the user is signed in."""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("ClimateLens")
        st.caption("Organizational Climate Assessment. Sign in to see your survey results.")
        sign_in_tab, register_tab = st.tabs(["Sign in", "Register"])
        with sign_in_tab:
            _sign_in_form()
        with register_tab:
            _register_form()
    st.stop()


def _sign_in_form() -> None:
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        try:
            session = client().auth.sign_in_with_password({"email": email.strip(), "password": password})
        except ValueError as error:
            st.error(str(error))
        except AuthError as error:
            # Only invalid_credentials is a wrong email/password; show anything else
            # (e.g. email_not_confirmed) as Supabase reports it
            if error.code in (None, "invalid_credentials"):
                st.error("Wrong email or password.")
            else:
                st.error(f"Sign-in failed: {error.message}")
        except Exception:
            st.error("Could not reach Supabase. Check the network and supabase_url.")
        else:
            _start_session(session.user)


def _register_form() -> None:
    with st.form("register"):
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", type="primary")

    if not submitted:
        return
    if not email.strip() or not password:
        st.error("Enter an email and a password.")
        return
    if password != confirm:
        st.error("The passwords do not match.")
        return

    try:
        response = client().auth.sign_up({"email": email.strip(), "password": password})
    except ValueError as error:
        st.error(str(error))
    except AuthError as error:
        # Supabase checks password strength, email format and duplicates itself
        if error.code == "signup_disabled":
            st.error("Registration is turned off for this project. Ask an admin for an account.")
        else:
            st.error(f"Registration failed: {error.message}")
    except Exception:
        st.error("Could not reach Supabase. Check the network and supabase_url.")
    else:
        # With email confirmation on, Supabase returns no session until the link is clicked.
        # It answers the same way for an email that is already registered, so say nothing more.
        if response.session is None:
            st.success(f"Check {email.strip()} for a confirmation link, then sign in.")
        else:
            _start_session(response.user)


def logout() -> None:
    try:
        client().auth.sign_out()
    except Exception:
        pass  # The local session is dropped below either way
    st.session_state.clear()
