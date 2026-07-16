"""Train and validate majority-class and TF-IDF Logistic Regression baselines."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cs_classifier.metrics import classification_metrics


def read_split(name: str) -> list[dict[str, str]]:
    with (PROJECT_ROOT / "data/processed" / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_pipeline(c_value: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("classifier", LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, random_state=42))])


def save_confusion_matrix(matrix: list[list[int]], labels: list[str], output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(np.array(matrix), cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(column, row, str(value), ha="center", va="center")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_errors(rows: list[dict[str, str]], predictions: np.ndarray, probabilities: np.ndarray, label_codes: list[str], path: Path) -> int:
    errors = []
    for row, predicted_id, probability_row in zip(rows, predictions, probabilities, strict=True):
        predicted_code = label_codes[int(predicted_id)]
        if row["label"] != predicted_code:
            errors.append({"id": row["id"], "text": row["text"], "true_label": row["label"], "predicted_label": predicted_code, "confidence": round(float(max(probability_row)), 6)})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "true_label", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(errors)
    return len(errors)


def main() -> None:
    labels_config = yaml.safe_load((PROJECT_ROOT / "configs/labels.yaml").read_text(encoding="utf-8"))
    label_codes = [item["code"] for item in labels_config["labels"]]
    label_to_id = {code: index for index, code in enumerate(label_codes)}
    label_ids = list(range(len(label_codes)))
    train_rows, validation_rows = read_split("train"), read_split("validation")
    train_texts = [row["text"] for row in train_rows]
    validation_texts = [row["text"] for row in validation_rows]
    train_y = [label_to_id[row["label"]] for row in train_rows]
    validation_y = [label_to_id[row["label"]] for row in validation_rows]
    majority_id = min(set(train_y), key=lambda label_id: (-train_y.count(label_id), label_id))
    majority_metrics = classification_metrics(validation_y, [majority_id] * len(validation_y), label_ids, label_codes)
    candidates = []
    for c_value in [0.1, 1.0, 3.0]:
        model = make_pipeline(c_value)
        model.fit(train_texts, train_y)
        candidates.append({"C": c_value, "macro_f1": classification_metrics(validation_y, model.predict(validation_texts).tolist(), label_ids, label_codes)["macro_f1"]})
    best_c = max(candidates, key=lambda item: (item["macro_f1"], -item["C"]))["C"]
    best_model = make_pipeline(best_c)
    best_model.fit(train_texts, train_y)
    predictions = best_model.predict(validation_texts)
    probabilities = best_model.predict_proba(validation_texts)
    tfidf_metrics = classification_metrics(validation_y, predictions.tolist(), label_ids, label_codes)
    metrics_dir, figures_dir, models_dir = PROJECT_ROOT / "artifacts/metrics", PROJECT_ROOT / "artifacts/figures", PROJECT_ROOT / "artifacts/models"
    for directory in [metrics_dir, figures_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_split": "validation", "selection_metric": "macro_f1", "majority_label": label_codes[majority_id],
        "tfidf_candidates": candidates, "selected_tfidf_C": best_c, "majority_class": majority_metrics,
        "tfidf_logistic_regression": tfidf_metrics,
        "tfidf_validation_errors": write_errors(validation_rows, predictions, probabilities, label_codes, metrics_dir / "tfidf-validation-errors.csv"),
    }
    (metrics_dir / "baseline-validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_confusion_matrix(majority_metrics["confusion_matrix"], label_codes, figures_dir / "majority-validation-confusion-matrix.png", "Majority baseline (validation)")
    save_confusion_matrix(tfidf_metrics["confusion_matrix"], label_codes, figures_dir / "tfidf-validation-confusion-matrix.png", "TF-IDF + Logistic Regression (validation)")
    joblib.dump(best_model, models_dir / "tfidf-logistic-regression.joblib")
    print(json.dumps({"majority_macro_f1": majority_metrics["macro_f1"], "tfidf_macro_f1": tfidf_metrics["macro_f1"], "selected_C": best_c}, ensure_ascii=False))


if __name__ == "__main__":
    main()
