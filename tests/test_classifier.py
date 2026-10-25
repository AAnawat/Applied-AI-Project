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


if __name__ == "__main__":
    unittest.main()
