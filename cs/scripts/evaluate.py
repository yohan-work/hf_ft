"""Final one-time Test evaluation for the selected baseline and fine-tuned Transformer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cs_classifier.metrics import classification_metrics


def read_test_rows() -> list[dict[str, str]]:
    with (PROJECT_ROOT / "data/processed/test.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path, title: str) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def model_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def transformer_probabilities(model, tokenizer, texts: list[str], max_length: int, device: torch.device) -> np.ndarray:
    results = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), 8):
            batch = tokenizer(texts[start : start + 8], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            results.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(results, axis=0)


def milliseconds_per_text(function, texts: list[str], repeats: int = 5) -> float:
    function(texts)  # warm-up
    started = time.perf_counter()
    for _ in range(repeats):
        function(texts)
    elapsed = time.perf_counter() - started
    return round((elapsed / (len(texts) * repeats)) * 1000, 3)


def collect_examples(rows: list[dict[str, str]], probabilities: np.ndarray, label_codes: list[str], model_name: str) -> dict[str, list[dict[str, object]]]:
    records = []
    for row, scores in zip(rows, probabilities, strict=True):
        predicted_id = int(np.argmax(scores))
        records.append(
            {
                "model": model_name,
                "id": row["id"],
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": label_codes[predicted_id],
                "confidence": round(float(scores[predicted_id]), 6),
                "is_error": row["label"] != label_codes[predicted_id],
            }
        )
    errors = [record for record in records if record["is_error"]]
    return {
        "all_errors": errors,
        "lowest_confidence": sorted(records, key=lambda item: item["confidence"])[:5],
        "highest_confidence_errors": sorted(errors, key=lambda item: item["confidence"], reverse=True)[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="allow overwriting a previous final Test report")
    args = parser.parse_args()
    metrics_path = PROJECT_ROOT / "artifacts/metrics/final-test-metrics.json"
    if metrics_path.exists() and not args.force:
        raise SystemExit("Final Test report already exists. Use --force only for an explicitly approved rerun.")

    config = yaml.safe_load((PROJECT_ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
    labels_config = yaml.safe_load((PROJECT_ROOT / "configs/labels.yaml").read_text(encoding="utf-8"))
    label_codes = [item["code"] for item in labels_config["labels"]]
    label_to_id = {code: index for index, code in enumerate(label_codes)}
    label_ids = list(range(len(label_codes)))
    rows = read_test_rows()
    texts = [row["text"] for row in rows]
    truth = [label_to_id[row["label"]] for row in rows]

    baseline = joblib.load(PROJECT_ROOT / "artifacts/models/tfidf-logistic-regression.joblib")
    baseline_probabilities = baseline.predict_proba(texts)
    baseline_predictions = np.argmax(baseline_probabilities, axis=1)
    baseline_metrics = classification_metrics(truth, baseline_predictions.tolist(), label_ids, label_codes)
    baseline_examples = collect_examples(rows, baseline_probabilities, label_codes, "tfidf_logistic_regression")
    baseline_time = milliseconds_per_text(lambda batch: baseline.predict_proba(batch), texts)

    model_path = PROJECT_ROOT / "models/klue-roberta-cs-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    transformer_scores = transformer_probabilities(model, tokenizer, texts, config["model"]["max_length"], device)
    transformer_predictions = np.argmax(transformer_scores, axis=1)
    transformer_metrics = classification_metrics(truth, transformer_predictions.tolist(), label_ids, label_codes)
    transformer_examples = collect_examples(rows, transformer_scores, label_codes, "fine_tuned_klue_roberta")
    transformer_time = milliseconds_per_text(
        lambda batch: transformer_probabilities(model, tokenizer, batch, config["model"]["max_length"], device), texts
    )

    metrics_dir = PROJECT_ROOT / "artifacts/metrics"
    figures_dir = PROJECT_ROOT / "artifacts/figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(baseline_metrics["confusion_matrix"], label_codes, figures_dir / "tfidf-test-confusion-matrix.png", "TF-IDF baseline (Test)")
    save_confusion_matrix(transformer_metrics["confusion_matrix"], label_codes, figures_dir / "transformer-test-confusion-matrix.png", "Fine-tuned Transformer (Test)")
    with (metrics_dir / "final-test-errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "id", "text", "true_label", "predicted_label", "confidence", "is_error"])
        writer.writeheader()
        writer.writerows(baseline_examples["all_errors"] + transformer_examples["all_errors"])
    payload = {
        "evaluation_split": "test",
        "test_size": len(rows),
        "label_order": label_codes,
        "tfidf_logistic_regression": {
            "metrics": baseline_metrics,
            "inference_ms_per_text": baseline_time,
            "model_size_bytes": model_size_bytes(PROJECT_ROOT / "artifacts/models/tfidf-logistic-regression.joblib"),
            **baseline_examples,
        },
        "fine_tuned_klue_roberta": {
            "metrics": transformer_metrics,
            "inference_ms_per_text": transformer_time,
            "model_size_bytes": model_size_bytes(model_path),
            "device": str(device),
            **transformer_examples,
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_macro_f1": baseline_metrics["macro_f1"], "transformer_macro_f1": transformer_metrics["macro_f1"], "transformer_errors": len(transformer_examples["all_errors"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
