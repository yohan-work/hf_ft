"""Run a single local prediction without starting the API server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cs_classifier.inference import ClassifierPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="customer inquiry to classify")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    predictor = ClassifierPredictor(PROJECT_ROOT / "models/klue-roberta-cs-v1", PROJECT_ROOT / "configs/labels.yaml", device=args.device)
    result = predictor.predict(args.text)
    print(json.dumps({"label": result.label, "label_name": result.label_name, "confidence": result.confidence, "scores": result.scores}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
