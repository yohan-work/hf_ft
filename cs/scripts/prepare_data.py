"""Create deterministic splits, run leakage checks, and inspect Hugging Face token lengths."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cs_classifier.data import SplitConfig, distribution, leakage_report, load_rows, split_rows, write_json, write_splits


def main() -> None:
    train_config = yaml.safe_load((PROJECT_ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
    label_config = yaml.safe_load((PROJECT_ROOT / "configs/labels.yaml").read_text(encoding="utf-8"))
    label_order = [item["code"] for item in label_config["labels"]]
    raw_path = PROJECT_ROOT / train_config["data"]["raw_path"]
    output_dir = PROJECT_ROOT / train_config["data"]["processed_dir"]
    rows = load_rows(raw_path, set(label_order))
    split_config = SplitConfig(
        seed=train_config["seed"],
        train_fraction=train_config["split"]["train"],
        validation_fraction=train_config["split"]["validation"],
        test_fraction=train_config["split"]["test"],
    )
    splits = split_rows(rows, split_config)
    report = leakage_report(splits)
    if not report["passed"]:
        raise RuntimeError(f"leakage check failed: {report}")
    write_splits(splits, output_dir)
    write_json(report, output_dir / "leakage-report.json")

    label_to_id = {label: index for index, label in enumerate(label_order)}
    dataset = DatasetDict(
        {
            name: Dataset.from_list([{**row, "label_id": label_to_id[row["label"]]} for row in rows_in_split])
            for name, rows_in_split in splits.items()
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(train_config["model"]["id"], revision=train_config["model"]["revision"])
    max_length = train_config["model"]["max_length"]
    token_lengths = {}
    for split_name, split_dataset in dataset.items():
        encoded = tokenizer(split_dataset["text"], truncation=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        token_lengths[split_name] = {
            "min": min(lengths),
            "max": max(lengths),
            "p95": sorted(lengths)[max(0, int(len(lengths) * 0.95) - 1)],
            "would_truncate_at_max_length": sum(length > max_length for length in lengths),
        }
    tokenized_dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        desc="Tokenizing CS inquiries",
    )
    tokenized_dataset.save_to_disk(str(output_dir / "hf_dataset"))
    summary = {
        "seed": train_config["seed"],
        "model_id": train_config["model"]["id"],
        "model_revision": train_config["model"]["revision"],
        "max_length": max_length,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "class_distribution": {name: distribution(rows, label_order) for name, rows in splits.items()},
        "label_to_id": label_to_id,
        "token_lengths": token_lengths,
        "dataset_features": str(dataset),
    }
    write_json(summary, output_dir / "data-summary.json")
    print(f"Wrote deterministic splits to {output_dir}")
    print(f"Split sizes: {summary['split_sizes']}")
    print(f"Token lengths: {summary['token_lengths']}")


if __name__ == "__main__":
    main()
