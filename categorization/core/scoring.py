"""Accuracy, F1 and latency for one categorizer's predictions."""

from sklearn.metrics import accuracy_score, classification_report, f1_score

from categorization.core.taxonomy import CODES

INVALID = "(invalid)"


def score(gold, predicted, seconds):
    """Invalid (missing) predictions count as wrong; per-class scores cover only codes in the answer key."""
    predicted = predicted.fillna(INVALID)
    labels = [code for code in CODES if code in set(gold)]
    report = classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0)
    return {
        "responses": len(gold),
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, predicted, labels=labels, average="weighted", zero_division=0),
        "invalid_rate": float((predicted == INVALID).mean()),
        "mean_seconds": float(seconds.mean()),
        "p95_seconds": float(seconds.quantile(0.95)),
        "per_class": {code: report[code] for code in labels},
    }
