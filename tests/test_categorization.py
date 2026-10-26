"""Offline tests. No Ollama server or model needed."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from categorization.core.exam import few_shot_overlap, join, load_answer_key, load_exam
from categorization.core.examples import FEW_SHOT_EXAMPLES
from categorization.core.llm import _post, build_messages, categorize, parse_category
from categorization.core.paths import EXAMS_DIR
from categorization.core.scoring import INVALID, score
from categorization.core.taxonomy import CODES


def fake_chat(replies):
    calls = []
    replies = iter(replies)

    def post(host, path, payload, timeout):
        calls.append(payload)
        return {"message": {"content": next(replies)}}

    return post, calls


def write_csv(rows):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(handle, index=False)
    handle.close()
    return Path(handle.name)


class ParseCategoryTest(unittest.TestCase):
    def test_valid_code(self):
        self.assertEqual(parse_category('{"category": "workload"}'), "workload")

    def test_rejects_unknown_code_and_bad_json(self):
        self.assertIsNone(parse_category('{"category": "salary"}'))
        self.assertIsNone(parse_category("workload"))
        self.assertIsNone(parse_category('["workload"]'))
        self.assertIsNone(parse_category('{"category": []}'))
        self.assertIsNone(parse_category('{"category": {}}'))


class CategorizeTest(unittest.TestCase):
    def test_marks_invalid_outputs(self):
        post, calls = fake_chat(['{"category": "compensation"}', "not json"])
        with patch("categorization.core.llm._post", post):
            result = categorize(pd.Series(["เงินเดือนน้อย", "อะไรก็ได้"], index=[5, 9]))

        self.assertEqual(list(result.index), [5, 9])
        self.assertEqual(result.loc[5, "category"], "compensation")
        self.assertTrue(pd.isna(result.loc[9, "category"]))
        self.assertEqual(list(result["category_valid"]), [True, False])
        self.assertEqual(calls[0]["format"]["properties"]["category"]["enum"], CODES)
        self.assertEqual(calls[0]["options"]["temperature"], 0)

    def test_reports_progress_after_each_row(self):
        post, _ = fake_chat(['{"category": "compensation"}', '{"category": "workload"}'])
        seen = []
        with patch("categorization.core.llm._post", post):
            categorize(["เงินเดือนน้อย", "งานเยอะ"], progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(1, 2), (2, 2)])

    def test_timeout_and_bad_server_responses_are_actionable(self):
        with patch("categorization.core.llm.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(RuntimeError, "Cannot reach Ollama"):
                _post("http://127.0.0.1:11434", "/api/chat", {}, 1)
        for response in [None, [], {"message": None}, {"message": {"content": []}}]:
            with self.subTest(response=response), patch(
                "categorization.core.llm._post", return_value=response
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid chat response"):
                    categorize(["test"])

    def test_few_shot_adds_example_turns(self):
        post, calls = fake_chat(['{"category": "other"}'])
        with patch("categorization.core.llm._post", post):
            categorize(["ทดสอบ"], few_shot=True)

        messages = calls[0]["messages"]
        self.assertEqual(len(messages), 2 + 2 * len(FEW_SHOT_EXAMPLES))
        self.assertEqual(messages[-1], {"role": "user", "content": "ทดสอบ"})
        self.assertEqual(json.loads(messages[2]["content"]), {"category": FEW_SHOT_EXAMPLES[0][1]})

    def test_rejects_blank_and_missing_text(self):
        with self.assertRaises(ValueError):
            categorize(["  "])
        with self.assertRaises(ValueError):
            categorize(pd.Series(["ok", None]))

    def test_zero_shot_prompt_lists_every_code(self):
        system = build_messages("x")[0]["content"]
        for code in CODES:
            self.assertIn(f"- {code}:", system)


class ExamTest(unittest.TestCase):
    def test_join_matches_rows_by_id_not_order(self):
        exam = load_exam(write_csv({"id": ["1", "2"], "feedback": ["โอทีเยอะ", "เงินเดือนน้อย"]}))
        key = load_answer_key(write_csv({"id": ["2", "1"], "category": ["compensation", "workload"]}))
        joined = join(exam, key).set_index("id")

        self.assertEqual(joined.loc["1", "category"], "workload")
        self.assertEqual(joined.loc["2", "category"], "compensation")

    def test_join_rejects_ids_missing_from_either_file(self):
        exam = load_exam(write_csv({"id": ["1", "2"], "feedback": ["a", "b"]}))
        key = load_answer_key(write_csv({"id": ["1", "3"], "category": ["other", "other"]}))
        with self.assertRaisesRegex(ValueError, "no answer: 2.*no exam row: 3"):
            join(exam, key)

    def test_loaders_reject_bad_files(self):
        with self.assertRaises(ValueError):
            load_answer_key(write_csv({"id": ["1"], "category": ["salary"]}))
        with self.assertRaises(ValueError):
            load_exam(write_csv({"id": ["1", "1"], "feedback": ["a", "b"]}))
        with self.assertRaises(ValueError):
            load_exam(write_csv({"id": ["1", "2"], "feedback": ["a", "a"]}))
        with self.assertRaises(ValueError):
            load_exam(write_csv({"feedback": ["a"]}))

    def test_committed_exams_are_valid(self):
        exams = sorted(EXAMS_DIR.glob("*.csv"))
        self.assertTrue(exams)
        for path in exams:
            with self.subTest(exam=path.name):
                exam = load_exam(path)
                self.assertEqual(few_shot_overlap(exam["feedback"]), [])

    def test_overlap_with_few_shot_examples(self):
        text = FEW_SHOT_EXAMPLES[0][0]
        self.assertEqual(few_shot_overlap(pd.Series([text, "อื่น ๆ"])), [text])


class ScoreTest(unittest.TestCase):
    def test_invalid_predictions_count_as_errors(self):
        gold = pd.Series(["workload", "workload", "compensation", "compensation"])
        predicted = pd.Series(["workload", None, "compensation", "workload"])
        metrics = score(gold, predicted, pd.Series([1.0, 1.0, 1.0, 1.0]))

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["invalid_rate"], 0.25)
        self.assertEqual(set(metrics["per_class"]), {"workload", "compensation"})
        self.assertNotIn(INVALID, metrics["per_class"])
        self.assertAlmostEqual(metrics["per_class"]["compensation"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
