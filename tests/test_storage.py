"""Offline tests. No Supabase credentials, model file or database writes needed."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from sentiment.storage import init_connection, save_predictions


class FakeSupabase:
    def __init__(self, fail=None, fail_cleanup=False):
        self.rows = {name: [] for name in (
            "survey_batches", "responses", "sentiment_predictions",
            "departments", "categories", "category_predictions",
        )}
        self.calls = []
        self.next_id = 1
        self.fail = fail
        self.fail_cleanup = fail_cleanup

    def table(self, name):
        return Query(self, name)


class Query:
    def __init__(self, client, table):
        self.client, self.name = client, table
        self.operation, self.payload, self.filters = None, None, {}

    def select(self, columns):
        self.operation = "select"
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def limit(self, count):
        return self

    def insert(self, payload):
        self.operation, self.payload = "insert", payload
        return self

    def upsert(self, payload, on_conflict):
        self.operation, self.payload = "upsert", payload
        self.conflict = on_conflict
        return self

    def update(self, payload):
        self.operation, self.payload = "update", payload
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def execute(self):
        db = self.client
        db.calls.append((self.name, self.operation, copy.deepcopy(self.payload), dict(self.filters)))
        if db.fail == (self.name, self.operation):
            raise RuntimeError("simulated private SDK details")
        if db.fail_cleanup and self.operation == "delete":
            raise RuntimeError("simulated cleanup outage")
        table = db.rows[self.name]
        matching = [row for row in table if all(row.get(k) == v for k, v in self.filters.items())]
        if self.operation == "select":
            data = matching
        elif self.operation in ("insert", "upsert"):
            old = None
            if self.operation == "upsert":
                old = next((r for r in table
                            if r.get(self.conflict) == self.payload[self.conflict]), None)
            if old is not None:
                old.update(self.payload)
                data = [old]
            else:
                row = {"id": db.next_id, **self.payload}
                db.next_id += 1
                table.append(row)
                data = [row]
        elif self.operation == "update":
            for row in matching:
                row.update(self.payload)
            data = matching
        elif self.operation == "delete":
            data = matching
            ids = {row["id"] for row in matching}
            db.rows[self.name] = [r for r in table if r["id"] not in ids]
            if self.name == "survey_batches":
                response_ids = {r["id"] for r in db.rows["responses"] if r["batch_id"] in ids}
                db.rows["responses"] = [r for r in db.rows["responses"] if r["id"] not in response_ids]
                for name in ("sentiment_predictions", "category_predictions"):
                    db.rows[name] = [r for r in db.rows[name] if r["response_id"] not in response_ids]
        else:
            raise AssertionError(f"Unsupported operation: {self.operation}")
        return SimpleNamespace(data=copy.deepcopy(data))


class StorageTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "feedback": ["ทีมช่วยเหลือดี", "วันนี้มีประชุม"],
            "department": ["IT", "IT"],
            "sentiment": ["Positive", "Neutral"],
            "confidence": [.9, .6],
            "category": ["teamwork", None],
            "category_confidence": [.8, None],
            "employee_id": ["must-not-send", "must-not-send"],
        })

    def test_save_links_results_and_reuses_reference(self):
        db = FakeSupabase()
        batch, count = save_predictions(self.frame(), batch_name="October", client=db)
        self.assertEqual(count, 2)
        self.assertEqual(db.rows["survey_batches"], [{"id": batch, "name": "October", "row_count": 2}])
        self.assertEqual(len(db.rows["departments"]), 1)
        self.assertEqual(len(db.rows["category_predictions"]), 1)
        self.assertEqual(len(db.rows["sentiment_predictions"]), 2)
        response_ids = {r["id"] for r in db.rows["responses"]}
        self.assertEqual(response_ids, {r["response_id"] for r in db.rows["sentiment_predictions"]})
        self.assertEqual(sum(c[:2] == ("departments", "select") for c in db.calls), 1)
        self.assertNotIn("must-not-send", repr(db.calls))

    def test_missing_department_and_category_are_optional(self):
        db = FakeSupabase()
        frame = self.frame().drop(columns=["department", "category", "category_confidence"])
        save_predictions(frame, client=db)
        self.assertEqual(db.rows["departments"], [])
        self.assertEqual(db.rows["category_predictions"], [])
        self.assertTrue(all(r["department_id"] is None for r in db.rows["responses"]))

    def test_failure_cleans_only_new_batch_and_preserves_shared_references(self):
        db = FakeSupabase(fail=("category_predictions", "insert"))
        db.rows["survey_batches"] = [{"id": 500, "name": "Existing", "row_count": 1}]
        db.rows["responses"] = [{"id": 501, "batch_id": 500}]
        db.rows["sentiment_predictions"] = [{"id": 502, "response_id": 501}]
        with self.assertRaisesRegex(RuntimeError, "Removed batch"):
            save_predictions(self.frame(), client=db)
        self.assertEqual([r["id"] for r in db.rows["survey_batches"]], [500])
        self.assertEqual([r["id"] for r in db.rows["responses"]], [501])
        self.assertEqual([r["id"] for r in db.rows["sentiment_predictions"]], [502])
        self.assertEqual(len(db.rows["departments"]), 1)
        self.assertEqual(len(db.rows["categories"]), 1)

    def test_cleanup_failure_reports_batch_for_manual_check(self):
        db = FakeSupabase(fail=("sentiment_predictions", "insert"), fail_cleanup=True)
        with self.assertRaisesRegex(RuntimeError, "batch 1 could not be confirmed") as caught:
            save_predictions(self.frame(), client=db)
        self.assertNotIn("private SDK", str(caught.exception))
        self.assertEqual(db.rows["survey_batches"][0]["row_count"], 0)

    def test_unconfirmed_batch_creation_does_not_delete_other_uploads(self):
        db = FakeSupabase(fail=("survey_batches", "insert"))
        with self.assertRaisesRegex(RuntimeError, "Check survey_batches"):
            save_predictions(self.frame(), client=db)
        self.assertFalse(any(c[1] == "delete" for c in db.calls))

    def test_finalization_failure_also_cleans_up(self):
        db = FakeSupabase(fail=("survey_batches", "update"))
        with self.assertRaisesRegex(RuntimeError, "Removed batch"):
            save_predictions(self.frame(), client=db)
        self.assertEqual(db.rows["survey_batches"], [])
        self.assertEqual(db.rows["responses"], [])

    def test_validation_happens_before_connecting(self):
        frame = self.frame()
        frame.loc[0, "confidence"] = 2
        with patch("sentiment.storage.init_connection") as connect:
            with self.assertRaises(ValueError):
                save_predictions(frame)
            connect.assert_not_called()

    def test_publishable_key_is_rejected_before_connecting(self):
        factory = MagicMock()
        sdk = SimpleNamespace(create_client=factory)
        with patch.dict("sys.modules", {"supabase": sdk}), patch(
            "sentiment.storage.st.secrets",
            {"supabase_url": "https://example.supabase.co", "supabase_key": "sb_publishable_test"},
        ):
            with self.assertRaisesRegex(ValueError, "Secret key"):
                init_connection.__wrapped__()
        factory.assert_not_called()

    def test_duplicate_file_is_a_separate_upload(self):
        db = FakeSupabase()
        first, _ = save_predictions(self.frame(), client=db)
        second, _ = save_predictions(self.frame(), client=db)
        self.assertNotEqual(first, second)
        self.assertEqual(len(db.rows["responses"]), 4)
        self.assertEqual(len(db.rows["departments"]), 1)


if __name__ == "__main__":
    unittest.main()
