"""Upload-page regression tests; all database writes and predictions are mocked."""

import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
import classifier

USER_CLIENT = object()  # stands in for the signed-in user's Supabase client


def upload_fixture(*args, **kwargs):
    file = BytesIO("feedback,department\nข้อความหนึ่ง,IT\nข้อความสอง,IT\n".encode("utf-8"))
    file.name = "test.csv"
    return file


class UploadTests(unittest.TestCase):
    def prediction(self):
        return pd.DataFrame({
            "sentiment": ["Positive", "Neutral"],
            "confidence": [.9, .7],
            "category": ["Communication", "Other"],
        })

    def test_save_button_persists_once_and_shows_batch_id(self):
        with patch("streamlit.file_uploader", side_effect=upload_fixture), patch.object(
            classifier, "classify", return_value=self.prediction()
        ), patch("auth.client", return_value=USER_CLIENT), patch(
            "sentiment.storage.save_predictions", return_value=(123, 2)
        ) as save:
            app = AppTest.from_file(str(ROOT / "dashboard/views/upload.py")).run(timeout=20)
            self.assertFalse(app.exception)
            save.assert_not_called()
            next(b for b in app.button if b.label == "Save to Supabase").click().run(timeout=20)
            self.assertFalse(app.exception)
            save.assert_called_once()
            self.assertEqual(len(save.call_args.args[0]), 2)
            self.assertIn("date", save.call_args.args[0].columns)
            self.assertEqual(save.call_args.kwargs["batch_name"], "test")
            self.assertIs(save.call_args.kwargs["client"], USER_CLIENT)
            self.assertTrue(any("123" in message.value for message in app.success))
            self.assertTrue(next(b for b in app.button if b.label == "Save to Supabase").disabled)
            app.run(timeout=20)
            save.assert_called_once()

    def test_database_error_is_visible_and_not_marked_saved(self):
        with patch("streamlit.file_uploader", side_effect=upload_fixture), patch.object(
            classifier, "classify", return_value=self.prediction()
        ), patch("auth.client", return_value=USER_CLIENT), patch(
            "sentiment.storage.save_predictions",
            side_effect=RuntimeError("Saving failed. Check database permissions."),
        ):
            app = AppTest.from_file(str(ROOT / "dashboard/views/upload.py")).run(timeout=20)
            next(b for b in app.button if b.label == "Save to Supabase").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any("Saving failed" in error.value for error in app.error))
            self.assertFalse(next(b for b in app.button if b.label == "Save to Supabase").disabled)
            self.assertEqual(app.session_state["saved_uploads"], {})


if __name__ == "__main__":
    unittest.main()
