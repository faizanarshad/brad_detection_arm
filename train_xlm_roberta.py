"""
Fine-tune XLM-RoBERTa for ADG_CODE classification from GOOD_NAME (multilingual; good for Armenian + Latin brands).
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from train_bilstm import compute_class_weights, load_labeled_data, set_seed


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_datasets(tokenizer, texts_train, texts_val, y_train, y_val, max_length: int):
    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds_tr = Dataset.from_dict({"text": texts_train, "labels": y_train.tolist()})
    ds_va = Dataset.from_dict({"text": texts_val, "labels": y_val.tolist()})
    ds_tr = ds_tr.map(tokenize_batch, batched=True, remove_columns=["text"])
    ds_va = ds_va.map(tokenize_batch, batched=True, remove_columns=["text"])
    return ds_tr, ds_va


def compute_metrics_builder():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float((preds == labels).mean()),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="XLM-RoBERTa ADG_CODE classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "brand_task.xlsx",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xlm-roberta-base",
        help="HF model id (e.g. xlm-roberta-base, xlm-roberta-large)",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "xlm_roberta_adg")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4, help="Lower if GPU/MPS runs OOM (try 2 + --grad-accum 4)")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If >0, overrides epochs (useful for smoke tests)",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (use if MPS/CUDA runs out of memory)",
    )
    parser.add_argument(
        "--metric-for-best",
        choices=("f1_weighted", "f1_macro", "accuracy"),
        default="f1_weighted",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    df, y, label_encoder = load_labeled_data(args.data)
    texts = df["GOOD_NAME"].tolist()
    try:
        idx_train, idx_val = train_test_split(
            np.arange(len(texts)),
            test_size=args.val_size,
            random_state=args.seed,
            stratify=y,
        )
    except ValueError:
        idx_train, idx_val = train_test_split(
            np.arange(len(texts)),
            test_size=args.val_size,
            random_state=args.seed,
        )

    texts_train = [texts[i] for i in idx_train]
    texts_val = [texts[i] for i in idx_val]
    y_train = y[idx_train]
    y_val = y[idx_val]

    num_classes = len(label_encoder.classes_)
    classes = label_encoder.classes_.tolist()
    id2label = {i: str(c) for i, c in enumerate(classes)}
    label2id = {str(c): i for i, c in enumerate(classes)}

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    ds_tr, ds_va = build_datasets(tokenizer, texts_train, texts_val, y_train, y_val, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    cw = None
    if not args.no_class_weight:
        cw = compute_class_weights(y_train, num_classes)

    if args.cpu:
        use_cuda = False
        use_mps = False
    else:
        use_cuda = torch.cuda.is_available()
        use_mps = bool(torch.backends.mps.is_available()) and not use_cuda

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_kw = (
        {"evaluation_strategy": "epoch"}
        if "evaluation_strategy" in ta_params
        else {"eval_strategy": "epoch"}
    )
    ta_common = dict(
        output_dir=str(args.out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best,
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_cuda,
        report_to="none",
        seed=args.seed,
        logging_steps=50,
        **eval_kw,
    )
    if "use_mps_device" in ta_params:
        ta_common["use_mps_device"] = use_mps
    if args.cpu and "no_cuda" in ta_params:
        ta_common["no_cuda"] = True
    if args.max_steps > 0:
        ta_common["max_steps"] = args.max_steps
    training_args = TrainingArguments(**ta_common)

    common = dict(
        model=model,
        args=training_args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(),
    )
    if args.no_class_weight:
        trainer = Trainer(**common)
    else:
        trainer = WeightedTrainer(cw, **common)
    trainer.train()

    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    meta = {
        "base_model": args.model,
        "max_length": args.max_length,
        "classes": classes,
        "metric_for_best": args.metric_for_best,
    }
    with open(args.out / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = trainer.evaluate()
    with open(args.out / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print("Eval:", metrics)
    print(f"Saved model + tokenizer to {args.out}")


if __name__ == "__main__":
    main()
