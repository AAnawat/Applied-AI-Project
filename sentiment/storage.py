import json
import math
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from supabase import Client


@st.cache_resource(show_spinner="Connecting to Supabase...")
def init_connection() -> "Client":
    from supabase import create_client

    try:
        url: str = st.secrets["supabase_url"]
        key: str = st.secrets["supabase_key"]
    except (FileNotFoundError, KeyError):
        raise ValueError(
            "Set supabase_url and supabase_key in .streamlit/secrets.toml "
            "(or in the Streamlit hosting Secrets settings). "
            "See .streamlit/secrets.toml.example."
        ) from None
    if not isinstance(url, str) or not isinstance(key, str) or not url.strip() or not key.strip():
        raise ValueError("supabase_url and supabase_key must be non-empty strings.")

    return create_client(url, key)


def _optional_text(value, field, number):
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(f"Row {number}: {field} must be text.")

    return value.strip() or None


def _confidence(value, field, number):
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Row {number}: {field} must be numeric.")

    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Row {number}: {field} must be between 0 and 1.")

    return value


def prepare_records(results):
    if not results.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")

    missing = {"feedback", "sentiment"} - set(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {', '.join(sorted(missing))}")

    supported = ["feedback", "sentiment", "confidence", "department", "category", "category_confidence"]
    frame = results[[column for column in supported if column in results.columns]]

    rows = json.loads(frame.to_json(orient="records", date_format="iso"))
    records = []

    for number, row in enumerate(rows, start=1):
        feedback = row["feedback"]
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError(f"Row {number}: feedback must be non-empty text.")

        label = row["sentiment"]
        if label not in {"Positive", "Neutral", "Negative"}:
            raise ValueError(f"Row {number}: invalid sentiment.")

        category = _optional_text(row.get("category"), "category", number)
        category_confidence = _confidence(row.get("category_confidence"), "category_confidence", number)
        if category is None and category_confidence is not None:
            raise ValueError(f"Row {number}: category_confidence requires a category.")

        records.append({
            "feedback_text": feedback,
            "department": _optional_text(row.get("department"), "department", number),
            "label": label,
            "confidence": _confidence(row.get("confidence"), "confidence", number),
            "category": category,
            "category_confidence": category_confidence,
        })
    return records


def _returned_id(response):
    rows = response.data
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("Expected exactly one database record.")

    record_id = rows[0].get("id")
    if type(record_id) is not int or record_id <= 0:
        raise RuntimeError("Database did not return a valid integer ID.")

    return record_id


def get_or_create_reference(supabase, table, column, value):
    if value is None:
        return None

    response = (
        supabase
        .table(table)
        .select("id")
        .eq(column, value)
        .limit(1)
        .execute()
    )
    if response.data:
        return _returned_id(response)

    response = (
        supabase
        .table(table)
        .upsert({column: value}, on_conflict=column)
        .execute()
    )
    return _returned_id(response)


def save_predictions(results, *, batch_name="Survey upload", client=None):
    if not isinstance(batch_name, str) or not batch_name.strip():
        raise ValueError("Batch name must be non-empty text.")

    records = prepare_records(results)
    if not records:
        raise ValueError("No prediction results to save.")

    supabase = client if client is not None else init_connection()
    batch_id = None
    department_ids = {}
    category_ids = {}

    try:
        batch_response = (
            supabase
            .table("survey_batches")
            .insert({"name": batch_name.strip(), "row_count": 0})
            .execute()
        )
        batch_id = _returned_id(batch_response)

        for row in records:
            department = row["department"]
            if department not in department_ids:
                department_ids[department] = get_or_create_reference(
                    supabase, "departments", "name", department
                )

            response = (
                supabase
                .table("responses")
                .insert({
                    "batch_id": batch_id,
                    "department_id": department_ids[department],
                    "feedback_text": row["feedback_text"],
                })
                .execute()
            )
            response_id = _returned_id(response)

            (
                supabase
                .table("sentiment_predictions")
                .insert({
                    "response_id": response_id,
                    "label": row["label"],
                    "confidence": row["confidence"],
                })
                .execute()
            )

            category = row["category"]
            if category is not None:
                if category not in category_ids:
                    category_ids[category] = get_or_create_reference(
                        supabase, "categories", "code", category
                    )
                (
                    supabase
                    .table("category_predictions")
                    .insert({
                        "response_id": response_id,
                        "category_id": category_ids[category],
                        "confidence": row["category_confidence"],
                    })
                    .execute()
                )

        saved = (
            supabase
            .table("survey_batches")
            .update({"row_count": len(records)})
            .eq("id", batch_id)
            .execute()
        )
        if _returned_id(saved) != batch_id or saved.data[0].get("row_count") != len(records):
            raise RuntimeError("Could not confirm the completed row count.")

    except Exception:
        if batch_id is None:
            raise RuntimeError(
                "Could not confirm creation of the survey batch. Check survey_batches "
                "before retrying; the request may have reached the database."
            ) from None

        try:
            deleted = (
                supabase
                .table("survey_batches")
                .delete()
                .eq("id", batch_id)
                .execute()
            )
            if _returned_id(deleted) != batch_id:
                raise RuntimeError("Could not confirm cleanup.")
        except Exception:
            raise RuntimeError(
                f"Saving failed and cleanup of batch {batch_id} could not be confirmed. "
                "Check this batch in Supabase before retrying. It may contain partial "
                "results with row_count=0."
            ) from None

        raise RuntimeError(
            f"Saving failed. Removed batch {batch_id} and its "
            "responses/predictions. Shared department/category references were kept. "
            "Check the connection, schema and backend key permissions."
        ) from None

    return batch_id, len(records)
