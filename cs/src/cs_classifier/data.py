"""Dataset validation, deterministic splitting, and leakage checks."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedShuffleSplit

from .preprocess import canonical_for_duplicate_check, normalize_text

REQUIRED_COLUMNS = ("id", "text", "label", "source", "is_synthetic", "review_status", "notes")


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def __post_init__(self) -> None:
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")


def load_rows(path: Path, valid_labels: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(f"unexpected columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("dataset is empty")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids detected")
    for row in rows:
        row["text"] = normalize_text(row["text"])
        if row["label"] not in valid_labels:
            raise ValueError(f"unknown label: {row['label']}")
        if any(not row[column].strip() for column in REQUIRED_COLUMNS):
            raise ValueError(f"blank required field in {row['id']}")
    return rows


def split_rows(rows: list[dict[str, str]], config: SplitConfig) -> dict[str, list[dict[str, str]]]:
    labels = [row["label"] for row in rows]
    indices = list(range(len(rows)))
    first = StratifiedShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
    train_validation_idx, test_idx = next(first.split(indices, labels))
    train_validation = [rows[index] for index in train_validation_idx]
    test = [rows[index] for index in test_idx]
    validation_relative_fraction = config.validation_fraction / (config.train_fraction + config.validation_fraction)
    second_labels = [row["label"] for row in train_validation]
    second_indices = list(range(len(train_validation)))
    second = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_relative_fraction,
        random_state=config.seed,
    )
    train_idx, validation_idx = next(second.split(second_indices, second_labels))
    return {
        "train": [train_validation[index] for index in train_idx],
        "validation": [train_validation[index] for index in validation_idx],
        "test": test,
    }


def _character_bigrams(text: str) -> set[str]:
    canonical = canonical_for_duplicate_check(text)
    return {canonical[index : index + 2] for index in range(len(canonical) - 1)}


def leakage_report(splits: dict[str, list[dict[str, str]]], similarity_threshold: float = 0.60) -> dict[str, Any]:
    all_rows = [(split, row) for split, rows in splits.items() for row in rows]
    canonical_rows = [(split, row, canonical_for_duplicate_check(row["text"])) for split, row in all_rows]
    exact_cross_split: list[dict[str, str]] = []
    similarity_cross_split: list[dict[str, Any]] = []
    for (left_split, left, left_text), (right_split, right, right_text) in combinations(canonical_rows, 2):
        if left_split == right_split:
            continue
        if left_text == right_text:
            exact_cross_split.append({"left_id": left["id"], "right_id": right["id"]})
            continue
        left_bigrams, right_bigrams = _character_bigrams(left["text"]), _character_bigrams(right["text"])
        union = left_bigrams | right_bigrams
        score = len(left_bigrams & right_bigrams) / len(union) if union else 0.0
        if score >= similarity_threshold:
            similarity_cross_split.append(
                {"left_id": left["id"], "right_id": right["id"], "score": round(score, 4)}
            )
    return {
        "similarity_method": "character_bigram_jaccard",
        "similarity_threshold": similarity_threshold,
        "exact_cross_split_duplicates": exact_cross_split,
        "high_similarity_cross_split_pairs": similarity_cross_split,
        "passed": not exact_cross_split and not similarity_cross_split,
    }


def distribution(rows: list[dict[str, str]], label_order: list[str]) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    return {label: counts[label] for label in label_order}


def write_splits(splits: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        with (output_dir / f"{split_name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
