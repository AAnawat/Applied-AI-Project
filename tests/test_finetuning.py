"""CPU-only checks: no GPU, model download or external data upload."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from categorization.finetuning.common import CODES, sha256, verify_manifest
from categorization.finetuning.prepare import load_labelled, prepare, split_rows
from categorization.finetuning.runtime import encode_example
from categorization.finetuning.evaluate import evaluate, load_external_csv, parse_output


def fixture():
    # Synthetic English strings ONLY for unit tests, never a training dataset.
    return [
        {"id": f"{label}-{group}-{variant}", "feedback": f"Test {label} family {group} version {variant}",
         "category": label, "group_id": f"{label}-{group}"}
        for label in CODES for group in range(6) for variant in range(2)
    ]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "feedback", "category", "group_id"])
        writer.writeheader()
        writer.writerows(rows)


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]


class FineTuningTests(unittest.TestCase):
    def test_paraphrase_groups_stay_together_and_every_class_is_present(self):
        splits = split_rows(fixture())
        owners = {}
        for name, rows in splits.items():
            self.assertEqual({row["category"] for row in rows}, set(CODES))
            for row in rows:
                self.assertEqual(owners.setdefault(row["group_id"], name), name)
        self.assertEqual(sum(map(len, splits.values())), len(fixture()))

    def test_pending_review_file_cannot_be_used_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_text(
                "id,feedback,category,group_id,reviewed_category,review_status\n"
                "1,new feedback,career,g1,,pending\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "human review is not complete"):
                load_labelled(path, set())

    def test_approved_reviewed_label_overrides_ai_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_text(
                "id,feedback,category,group_id,reviewed_category,review_status\n"
                "1,new feedback,career,g1,communication,approved\n", encoding="utf-8"
            )
            rows = load_labelled(path, set())
            self.assertEqual(rows[0]["category"], "communication")

    def test_split_is_reproducible(self):
        self.assertEqual(split_rows(fixture(), 42), split_rows(fixture(), 42))

    def test_other_is_required(self):
        with self.assertRaisesRegex(ValueError, "other"):
            split_rows([row for row in fixture() if row["category"] != "other"])

    def test_conflicting_family_labels_are_rejected(self):
        rows = fixture()
        rows[1] = {**rows[1], "category": "other"}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            split_rows(rows)

    def test_protected_development_text_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_csv(path, [{"id": "1", "feedback": "Old exam text", "category": "career", "group_id": "1"}])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_labelled(path, {"oldexamtext"})

    def test_normalized_duplicate_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            rows = [
                {"id": "1", "feedback": "ABC text", "category": "career", "group_id": "1"},
                {"id": "2", "feedback": "ＡＢＣ  text", "category": "career", "group_id": "2"},
            ]
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_labelled(path, set())

    def test_manifest_detects_changed_data_and_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "input.csv", Path(directory) / "prepared"
            write_csv(path, fixture())
            with patch("categorization.finetuning.prepare.protected_feedback", return_value=set()):
                prepare(path, output)
                self.assertEqual(verify_manifest(output)["categories"], CODES)
                with self.assertRaises(FileExistsError):
                    prepare(path, output)
            with (output / "train.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "edited"):
                verify_manifest(output)

    def test_loss_masks_prompt_but_trains_json_and_eos(self):
        row = {"id": "one", "messages": [
            {"role": "user", "content": "text"},
            {"role": "assistant", "content": '{"category":"career"}'},
        ]}
        encoded = encode_example(row, FakeTokenizer(), 200)
        self.assertEqual(encoded["labels"][:6], [-100] * 6)
        self.assertEqual(encoded["labels"][6:], encoded["input_ids"][6:])
        self.assertEqual(encoded["labels"][-1], 999)
        with self.assertRaisesRegex(ValueError, "never silently truncated"):
            encode_example(row, FakeTokenizer(), 10)

    def test_external_csv_uses_reviewed_label_without_claiming_human_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "challenge.csv"
            path.write_text(
                "id,feedback,category,reviewed_category,review_status\n"
                "1,work computer is broken,workload,environment,reviewed_ai\n",
                encoding="utf-8-sig",
            )
            rows, statuses = load_external_csv(path)
            self.assertEqual(rows[0]["category"], "environment")
            self.assertEqual(statuses, ["reviewed_ai"])
            self.assertEqual(rows[0]["messages"][-2]["content"], "work computer is broken")
            self.assertNotIn("reviewed_ai", rows[0]["messages"][-2]["content"])

    def test_external_csv_rejects_missing_gold_invalid_labels_and_duplicates(self):
        cases = [
            "id,feedback\n1,text\n",
            "id,feedback,category,reviewed_category\n1,text,career,\n",
            "id,feedback,category\n1,text,unknown\n",
            "id,feedback,category\n1,text,career\n1,another,career\n",
            "id,feedback,category\n1,ABC text,career\n2,ＡＢＣ  text,career\n",
            "id,feedback,category\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "challenge.csv"
            for content in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_external_csv(path)

    def test_external_csv_cannot_be_marked_as_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text("{}")
            args = SimpleNamespace(
                run=root, input_csv=root / "challenge.csv",
                split="test", confirm_final_test=True,
            )
            with patch("categorization.finetuning.evaluate.cuda_runtime") as gpu:
                with self.assertRaisesRegex(ValueError, "development-only"):
                    evaluate(args)
                gpu.assert_not_called()

    def test_bad_json_category_is_invalid(self):
        self.assertEqual(parse_output('{"category":"career"}'), "career")
        self.assertIsNone(parse_output('{"category":[]}'))
        self.assertIsNone(parse_output('{"category":"new-topic"}'))
        self.assertIsNone(parse_output("not json"))

    def test_final_test_requires_explicit_confirmation_before_gpu_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            run = root / "run"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({
                "data_manifest_sha256": sha256(root / "manifest.json"), "smoke_only": False,
            }))
            args = SimpleNamespace(data_dir=root, run=run, split="test", confirm_final_test=False)
            with patch("categorization.finetuning.evaluate.verify_manifest", return_value={}), patch(
                "categorization.finetuning.evaluate.cuda_runtime"
            ) as gpu:
                with self.assertRaisesRegex(ValueError, "confirm-final-test"):
                    evaluate(args)
                gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
