"""Load a saved classifier once and return label probabilities for a text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class PredictionResult:
    label: str
    label_name: str
    confidence: float
    scores: list[dict[str, float | str]]


def select_device(requested: str = "auto") -> torch.device:
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        requested = "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    if requested not in available or not available[requested]:
        raise ValueError(f"requested device is unavailable: {requested}")
    return torch.device(requested)


class ClassifierPredictor:
    """Inference wrapper; load this once per API process, never per request."""

    def __init__(self, model_path: Path, labels_path: Path, max_length: int = 128, device: str = "auto") -> None:
        labels_config = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
        self.label_names = {item["code"]: item["name_ko"] for item in labels_config["labels"]}
        self.max_length = max_length
        self.device = select_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)[0].tolist()
        def label_for_index(index: int) -> str:
            id2label = self.model.config.id2label
            return str(id2label[index] if index in id2label else id2label[str(index)])

        ranked = sorted(
            [
                {
                    "label": label_for_index(index),
                    "score": round(float(score), 6),
                }
                for index, score in enumerate(probabilities)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        best = ranked[0]
        return PredictionResult(
            label=str(best["label"]),
            label_name=self.label_names[str(best["label"])],
            confidence=float(best["score"]),
            scores=ranked,
        )
