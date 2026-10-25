"""Login gate and Supabase loading tests; no Supabase project needed."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

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


class LoginGateTests(unittest.TestCase):
    def test_signed_out_visitor_only_sees_the_login_form(self):
        app = AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual([b.label for b in app.button], ["Sign in"])
        self.assertEqual(len(app.text_input), 2)
        self.assertEqual(len(app.sidebar), 0)


if __name__ == "__main__":
    unittest.main()
