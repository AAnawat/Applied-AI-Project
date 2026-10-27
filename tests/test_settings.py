"""Settings-page tests; the Ollama server is mocked."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
import classifier

PAGE = str(ROOT / "dashboard/views/settings.py")


def settings_page(allow=True):
    app = AppTest.from_file(PAGE)
    app.secrets["ollama_host"] = "http://ollama:11434"
    app.secrets["allow_llm_override"] = allow
    return app.run(timeout=20)


def button(app, label):
    return next(b for b in app.button if b.label == label)


class SettingsTests(unittest.TestCase):
    def test_page_is_inert_unless_secrets_allow_it(self):
        app = settings_page(allow=False)
        self.assertFalse(app.exception)
        self.assertIn("turned off", app.info[0].value)
        self.assertEqual(len(app.text_input), 0)

    def test_checked_server_and_model_are_saved_for_the_session_and_can_be_reset(self):
        with patch("categorization.core.llm.list_models", return_value=["typhoon:4b", "finetuned:4b"]) as models:
            app = settings_page()
            self.assertFalse(app.exception)
            self.assertIn("Default server", app.markdown[0].value)
            self.assertFalse(any(b.label == "Save" for b in app.button))

            app.text_input[0].input("http://10.110.192.234:11434/").run(timeout=20)
            button(app, "Check server").click().run(timeout=20)
            models.assert_called_once_with("http://10.110.192.234:11434")
            self.assertEqual(app.selectbox[0].options, ["typhoon:4b", "finetuned:4b"])

            app.selectbox[0].select("finetuned:4b").run(timeout=20)
            button(app, "Save").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state[classifier.OVERRIDE_KEY], {
                "host": "http://10.110.192.234:11434", "model": "finetuned:4b", "timeout": 120.,
            })
            self.assertIn("Saved", app.success[0].value)
            self.assertIn("Custom server", app.markdown[0].value)

            button(app, "Reset to default").click().run(timeout=20)
            self.assertNotIn(classifier.OVERRIDE_KEY, app.session_state)
            self.assertIn("Default server", app.markdown[0].value)

    def test_unreachable_or_internet_servers_cannot_be_saved(self):
        with patch("categorization.core.llm.list_models", side_effect=RuntimeError("Cannot reach Ollama")):
            app = settings_page()
            app.text_input[0].input("http://10.0.0.9:11434").run(timeout=20)
            button(app, "Check server").click().run(timeout=20)
            self.assertIn("Cannot reach Ollama", app.error[0].value)
            self.assertFalse(any(b.label == "Save" for b in app.button))

        with patch("categorization.core.llm.list_models") as models:
            app = settings_page()
            app.text_input[0].input("http://8.8.8.8:11434").run(timeout=20)
            button(app, "Check server").click().run(timeout=20)
            self.assertIn("not on the local network", app.error[0].value)
            models.assert_not_called()
            self.assertFalse(any(b.label == "Save" for b in app.button))
        self.assertNotIn(classifier.OVERRIDE_KEY, app.session_state)

    def test_server_without_models_is_not_saveable(self):
        with patch("categorization.core.llm.list_models", return_value=[]):
            app = settings_page()
            app.text_input[0].input("http://10.0.0.9:11434").run(timeout=20)
            button(app, "Check server").click().run(timeout=20)
            self.assertIn("no models", app.warning[0].value)
            self.assertFalse(any(b.label == "Save" for b in app.button))


if __name__ == "__main__":
    unittest.main()
