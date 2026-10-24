from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment.joblib"
LABEL_NAMES = {"pos": "Positive", "neu": "Neutral", "neg": "Negative"}


def load_model(path=DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Place the exported TF-IDF + classifier "
            "pipeline here, or provide its path with --model."
        )
    return joblib.load(path)


def predict_sentiment(texts, model):

    if isinstance(texts, pd.Series):
        series = texts.copy()
    else:
        series = pd.Series(texts, dtype=object)

    if series.empty:
        return pd.DataFrame(index=series.index, columns=["sentiment", "confidence"])
    if series.isna().any():
        raise ValueError("Feedback contains missing values; remove them before prediction.")

    cleaned = series.astype(str).str.strip()

    if cleaned.eq("").any():
        raise ValueError("Feedback contains blank text; remove it before prediction.")

    labels = model.predict(cleaned.tolist())
    probabilities = model.predict_proba(cleaned.tolist())

    return pd.DataFrame({
        "sentiment": [LABEL_NAMES[label] for label in labels],
        "confidence": probabilities.max(axis=1),
    }, index=series.index)
