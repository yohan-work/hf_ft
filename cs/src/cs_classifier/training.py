"""Shared training helpers for Hugging Face sequence classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import EvalPrediction

from .metrics import classification_metrics


def make_compute_metrics(label_codes: list[str]):
    label_ids = list(range(len(label_codes)))

    def compute_metrics(prediction: EvalPrediction) -> dict[str, Any]:
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        predicted_ids = np.argmax(logits, axis=-1).tolist()
        return classification_metrics(prediction.label_ids.tolist(), predicted_ids, label_ids, label_codes)

    return compute_metrics
