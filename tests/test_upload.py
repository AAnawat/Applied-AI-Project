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
            "category": ["communication", "other"],
        })

    def test_save_button_persists_once_and_shows_batch_id(self):
        with patch("streamlit.file_uploader", side_effect=upload_fixture), patch.object(
            classifier, "classify", return_value=self.prediction()
        ) as classify, patch("auth.client", return_value=USER_CLIENT), patch(
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
            classify.assert_called_once()
            self.assertEqual(save.call_args.args[0]["category"].tolist(), ["communication", "other"])

    def test_llm_failure_stops_before_save_or_dashboard_actions(self):
        with patch("streamlit.file_uploader", side_effect=upload_fixture), patch.object(
            classifier, "classify",
            side_effect=classifier.TopicCategorizationError("Start Ollama and pull this model."),
        ), patch("sentiment.storage.save_predictions") as save:
            app = AppTest.from_file(str(ROOT / "dashboard/views/upload.py")).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertIn("Start Ollama", app.error[0].value)
            self.assertFalse(any(b.label == "Save to Supabase" for b in app.button))
            self.assertNotIn("upload_analysis", app.session_state)
            save.assert_not_called()

    def test_new_file_is_analyzed_again_but_typing_batch_name_reuses_results(self):
        with patch("streamlit.file_uploader", side_effect=upload_fixture), patch.object(
            classifier, "classify", return_value=self.prediction()
        ) as classify:
            app = AppTest.from_file(str(ROOT / "dashboard/views/upload.py")).run(timeout=20)
            app.text_input[0].input("New batch name").run(timeout=20)
            self.assertEqual(classify.call_count, 1)

            def changed_file(*args, **kwargs):
                file = upload_fixture()
                content = BytesIO(file.getvalue().replace("ข้อความหนึ่ง".encode(), "ข้อความใหม่".encode()))
                content.name = file.name
                return content

            with patch("streamlit.file_uploader", side_effect=changed_file):
                app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(classify.call_count, 2)

    def test_prelabelled_category_cannot_bypass_local_llm(self):
        def prelabelled(*args, **kwargs):
            file = BytesIO("feedback,category\nข้อความ,wrong\n".encode())
            file.name = "prelabelled.csv"
            return file

        with patch("streamlit.file_uploader", side_effect=prelabelled), patch.object(
            classifier, "classify"
        ) as classify:
            app = AppTest.from_file(str(ROOT / "dashboard/views/upload.py")).run(timeout=20)
            self.assertIn("remove result columns", app.error[0].value)
            classify.assert_not_called()

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
