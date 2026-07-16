"""Fine-tune klue/roberta-base for Korean CS label classification on Train/Validation only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cs_classifier.training import make_compute_metrics


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
    labels_config = yaml.safe_load((PROJECT_ROOT / "configs/labels.yaml").read_text(encoding="utf-8"))
    label_codes = [item["code"] for item in labels_config["labels"]]
    label_to_id = {label: index for index, label in enumerate(label_codes)}
    training = config["training"]
    set_seed(training["seed"])
    model_id, revision = config["model"]["id"], config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=len(label_codes),
        id2label={index: label for index, label in enumerate(label_codes)},
        label2id=label_to_id,
        use_safetensors=True,
    )
    dataset = load_from_disk(str(PROJECT_ROOT / config["data"]["processed_dir"] / "hf_dataset"))
    keep_columns = {"input_ids", "attention_mask", "token_type_ids", "label_id"}
    removable = [column for column in dataset["train"].column_names if column not in keep_columns]
    dataset = dataset.remove_columns(removable).rename_column("label_id", "labels")
    mps_available = torch.backends.mps.is_available()
    checkpoint_dir = PROJECT_ROOT / "artifacts/checkpoints/klue-roberta-cs-v1"
    run_dir = PROJECT_ROOT / "artifacts/runs/klue-roberta-cs-v1"
    model_dir = PROJECT_ROOT / "models/klue-roberta-cs-v1"
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=training["metric_for_best_model"],
        greater_is_better=True,
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["train_batch_size"],
        per_device_eval_batch_size=training["eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        num_train_epochs=training["epochs"],
        weight_decay=training["weight_decay"],
        logging_dir=str(run_dir),
        logging_strategy="steps",
        logging_steps=5,
        report_to="none",
        seed=training["seed"],
        data_seed=training["seed"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        save_safetensors=True,
        use_cpu=not mps_available,
        use_mps_device=mps_available,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=make_compute_metrics(label_codes),
    )
    train_result = trainer.train()
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    reloaded_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
    reloaded_model.eval()
    sample_text = "결제는 완료됐는데 주문 내역이 보이지 않아요."
    encoded = reloaded_tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=config["model"]["max_length"])
    with torch.no_grad():
        probabilities = torch.softmax(reloaded_model(**encoded).logits, dim=-1)[0].tolist()
    top_id = int(np.argmax(probabilities))
    metrics_dir = PROJECT_ROOT / "artifacts/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "device": "mps" if mps_available else "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dataset_version": "synthetic_cs_v1",
        "seed": training["seed"],
        "hyperparameters": training,
        "trainable_parameters": model_parameter_count(model),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "sample_reload_prediction": {
            "text": sample_text,
            "label": label_codes[top_id],
            "confidence": round(float(probabilities[top_id]), 6),
        },
        "test_evaluated": False,
    }
    (metrics_dir / "transformer-validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_macro_f1": validation_metrics["eval_macro_f1"], "best_checkpoint": trainer.state.best_model_checkpoint, "sample": payload["sample_reload_prediction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
