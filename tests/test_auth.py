"""Login gate and Supabase loading tests; no Supabase project needed."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest
from supabase_auth.errors import AuthApiError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
import climate


class FakeView:
    """Serves rows like PostgREST: at most PAGE_SIZE per .range() request."""

    def __init__(self, rows):
        self.rows, self.requests = rows, []

    def table(self, name):
        self.name = name
        return self

    def select(self, columns):
        return self

    def order(self, column):
        return self

    def range(self, start, end):
        self.requests.append((start, end))
        self.page = self.rows[start:min(end + 1, start + climate.PAGE_SIZE)]
        return self

    def execute(self):
        return SimpleNamespace(data=self.page)


def view_row(i):
    return {"response_id": i, "month": "2026-10", "department": "IT", "feedback": "ok",
            "sentiment": "Neutral", "category": None, "confidence": .5, "date": "2026-10-25"}


class LoadTests(unittest.TestCase):
    def setUp(self):
        climate.load_saved_responses.clear()

    def test_reads_every_page(self):
        view = FakeView([view_row(i) for i in range(climate.PAGE_SIZE + 5)])
        df = climate.load_saved_responses(view, "user-a")
        self.assertEqual(view.name, "v_analyzed_responses")
        self.assertEqual(len(df), climate.PAGE_SIZE + 5)
        self.assertEqual(len(view.requests), 2)
        self.assertEqual(set(df["category"]), {"Uncategorized"})

    def test_no_rows_gives_empty_frame_with_dashboard_columns(self):
        df = climate.load_saved_responses(FakeView([]), "user-b")
        self.assertTrue(df.empty)
        self.assertIn("sentiment", df.columns)


class FakeClient(FakeView):
    """A Supabase client whose sign_up answers like a project with or without email confirmation."""

    def __init__(self, confirm_email=True, error=None):
        super().__init__([])
        self.signups = []
        self.auth = SimpleNamespace(sign_up=self.sign_up)
        self.confirm_email, self.error = confirm_email, error

    def sign_up(self, credentials):
        self.signups.append(credentials)
        if self.error:
            raise self.error
        user = SimpleNamespace(id="new-user", email=credentials["email"])
        return SimpleNamespace(user=user, session=None if self.confirm_email else object())


def login_page(client=None):
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"))
    if client is not None:
        app.session_state["supabase"] = client
    return app.run(timeout=20)


def register(app, email, password, confirm=None):
    app.text_input(key="register_email").input(email)
    app.text_input(key="register_password").input(password)
    app.text_input[4].input(password if confirm is None else confirm)
    return app.button[1].click().run(timeout=20)


class LoginGateTests(unittest.TestCase):
    def test_signed_out_visitor_only_sees_sign_in_and_register(self):
        app = login_page()
        self.assertFalse(app.exception)
        self.assertEqual([t.label for t in app.tabs], ["Sign in", "Register"])
        self.assertEqual([b.label for b in app.button], ["Sign in", "Create account"])
        self.assertEqual(len(app.text_input), 5)
        self.assertEqual(len(app.sidebar), 0)


class RegisterTests(unittest.TestCase):
    def test_mismatched_passwords_are_not_sent(self):
        fake = FakeClient()
        app = register(login_page(fake), "new@test.com", "secret123", "secret456")
        self.assertEqual(app.error[0].value, "The passwords do not match.")
        self.assertEqual(fake.signups, [])

    def test_confirmation_required_keeps_the_user_signed_out(self):
        fake = FakeClient(confirm_email=True)
        app = register(login_page(fake), " new@test.com ", "secret123")
        self.assertEqual(fake.signups, [{"email": "new@test.com", "password": "secret123"}])
        self.assertIn("confirmation link", app.success[0].value)
        self.assertNotIn("user", app.session_state)
        self.assertEqual(len(app.sidebar), 0)

    def test_auto_confirmed_signup_opens_the_dashboard(self):
        fake = FakeClient(confirm_email=False)
        app = register(login_page(fake), "new@test.com", "secret123")
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["user"], {"id": "new-user", "email": "new@test.com"})
        self.assertIn("Signed in as **new@test.com**", [c.value for c in app.sidebar.caption])

    def test_disabled_signups_explain_what_to_do(self):
        fake = FakeClient(error=AuthApiError("Signups not allowed for this instance", 422, "signup_disabled"))
        app = register(login_page(fake), "new@test.com", "secret123")
        self.assertIn("Registration is turned off", app.error[0].value)

    def test_supabase_rejections_are_shown(self):
        fake = FakeClient(error=AuthApiError("Password should be at least 6 characters.", 422, "weak_password"))
        app = register(login_page(fake), "new@test.com", "abc")
        self.assertEqual(app.error[0].value, "Registration failed: Password should be at least 6 characters.")


if __name__ == "__main__":
    unittest.main()
