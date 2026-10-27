"""Dashboard-to-Ollama contract tests; no live model or database required."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
import classifier


class ClassifierTests(unittest.TestCase):
    def sentiment(self):
        return pd.DataFrame({
            "sentiment": ["Negative", "Positive"], "confidence": [.9, .8],
        }, index=[4, 8])

    def test_real_categorizer_contract_uses_typhoon_zero_shot_and_canonical_codes(self):
        replies = [
            {"message": {"content": '{"category": "workload"}'}},
            {"message": {"content": '{"category": "environment"}'}},
        ]
        with patch.object(classifier, "cached_sentiment_model", return_value=object()), patch.object(
            classifier, "predict_sentiment", return_value=self.sentiment()
        ), patch("categorization.core.llm._post", side_effect=replies) as post:
            result = classifier.classify(
                pd.Series(["งานเยอะจนไม่มีเวลาพัก", "คอมใหม่ใช้งานดี"], index=[4, 8])
            )
        self.assertEqual(result.index.tolist(), [4, 8])
        self.assertEqual(result["category"].tolist(), ["workload", "environment"])
        self.assertEqual(result["confidence"].tolist(), [.9, .8])
        self.assertNotIn("category_confidence", result.columns)
        payload = post.call_args_list[0].args[2]
        self.assertEqual(payload["model"], classifier.DEFAULT_MODEL)
        self.assertIn("typhoon2.5-qwen3-4b", payload["model"])
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertFalse(payload["think"])

    def test_invalid_output_aborts_instead_of_inventing_other_or_keyword_category(self):
        with patch.object(classifier, "cached_sentiment_model"), patch.object(
            classifier, "predict_sentiment", return_value=self.sentiment()
        ), patch("categorization.core.llm._post", return_value={"message": {"content": "not JSON"}}):
            with self.assertRaisesRegex(classifier.TopicCategorizationError, "invalid topic"):
                classifier.classify(pd.Series(["เงินเดือนน้อย", "หัวหน้าดี"], index=[4, 8]))

    def test_connection_failure_has_actionable_error(self):
        with patch.object(classifier, "cached_sentiment_model"), patch.object(
            classifier, "predict_sentiment", return_value=self.sentiment()
        ), patch("categorization.core.llm._post", side_effect=RuntimeError("unreachable")):
            with self.assertRaisesRegex(classifier.TopicCategorizationError, "Start Ollama"):
                classifier.classify(pd.Series(["เงินเดือนน้อย", "หัวหน้าดี"], index=[4, 8]))

    def test_server_configuration_is_read_without_touching_supabase(self):
        with patch("classifier.st.secrets", {
            "ollama_host": "http://llm-server:11434/",
            "ollama_model": "custom:4b",
            "ollama_timeout": 60,
        }):
            self.assertEqual(classifier.topic_settings(), {
                "host": "http://llm-server:11434", "model": "custom:4b", "timeout": 60.,
            })
        with patch("classifier.st.secrets", {}):
            self.assertEqual(classifier.topic_settings()["model"], classifier.DEFAULT_MODEL)
        for settings in [
            {"ollama_model": ""}, {"ollama_host": "server-without-scheme"},
            {"ollama_timeout": 0}, {"ollama_timeout": "nan"},
        ]:
            with self.subTest(settings=settings), patch("classifier.st.secrets", settings):
                with self.assertRaises(classifier.TopicCategorizationError):
                    classifier.topic_settings()

    def test_settings_page_server_is_used_only_when_secrets_allow_it(self):
        override = {classifier.OVERRIDE_KEY: {
            "host": "http://10.110.192.234:11434", "model": "custom:4b", "timeout": 30.,
        }}
        secrets = {"ollama_host": "http://ollama:11434"}
        with patch("classifier.st.secrets", secrets), patch("classifier.st.session_state", override):
            self.assertEqual(classifier.topic_settings()["host"], "http://ollama:11434")
            self.assertIsNone(classifier.topic_override())
        with patch("classifier.st.secrets", {**secrets, "allow_llm_override": True}), patch(
            "classifier.st.session_state", override
        ):
            self.assertEqual(classifier.topic_settings(), {
                "host": "http://10.110.192.234:11434", "model": "custom:4b", "timeout": 30.,
            })
        with patch("classifier.st.secrets", {**secrets, "allow_llm_override": "yes"}), patch(
            "classifier.st.session_state", override
        ):
            self.assertIsNone(classifier.topic_override())
        with patch("classifier.st.secrets", {**secrets, "allow_llm_override": True}), patch(
            "classifier.st.session_state", {}
        ):
            self.assertEqual(classifier.topic_settings()["host"], "http://ollama:11434")

    def test_settings_page_servers_must_be_on_the_local_network(self):
        for host in ["http://10.110.192.234:11434", "http://192.168.1.20:11434", "http://127.0.0.1:11434",
                     "http://[::1]:11434"]:
            with self.subTest(host=host):
                classifier.check_local_host(host)
        for host in ["http://8.8.8.8:11434", "http://169.254.169.254", "http://100.64.0.1:11434"]:
            with self.subTest(host=host), self.assertRaisesRegex(
                classifier.TopicCategorizationError, "not on the local network"
            ):
                classifier.check_local_host(host)
        # A name is judged by every address it resolves to
        resolved = [(None, None, None, "", ("10.0.0.5", 0)), (None, None, None, "", ("93.184.216.34", 0))]
        with patch("classifier.socket.getaddrinfo", return_value=resolved):
            with self.assertRaisesRegex(classifier.TopicCategorizationError, "not on the local network"):
                classifier.check_local_host("http://my-pc.example:11434")
        with patch("classifier.socket.getaddrinfo", side_effect=classifier.socket.gaierror):
            with self.assertRaisesRegex(classifier.TopicCategorizationError, "Cannot find"):
                classifier.check_local_host("http://no-such-pc:11434")

        override = {classifier.OVERRIDE_KEY: {"host": "http://8.8.8.8:11434", "model": "m", "timeout": 30.}}
        with patch("classifier.st.secrets", {"allow_llm_override": True}), patch(
            "classifier.st.session_state", override
        ):
            with self.assertRaises(classifier.TopicCategorizationError):
                classifier.topic_settings()


if __name__ == "__main__":
    unittest.main()
