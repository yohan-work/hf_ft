"""Classification metrics shared by baselines and transformers."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: list[int], y_pred: list[int], label_ids: list[int], label_codes: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=label_ids, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_f1": round(float(macro_f1), 6),
        "per_class": {
            code: {"precision": round(float(p), 6), "recall": round(float(r), 6), "f1": round(float(f), 6), "support": int(s)}
            for code, p, r, f, s in zip(label_codes, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
    }
