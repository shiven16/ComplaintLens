#!/usr/bin/env python3
"""Fine-tune a small DistilBERT baseline on a seeded training subset."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATA = Path("data/processed")
OUT = Path("models/distilbert")
REPORT = Path("reports/distilbert_metrics.csv")


class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(value[i]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def load(name, task):
    cols = ["narrative"] if task == "priority" else ["narrative", task]
    if task == "priority": cols += ["timely", "company_response"]
    data = pq.read_table(DATA / f"{name}.parquet", columns=cols).to_pydict()
    texts = [t or "" for t in data["narrative"]]
    labels = [int(t == "No" or "monetary relief" in (r or "").casefold())
              for t, r in zip(data["timely"], data["company_response"])] if task == "priority" else data[task]
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["product", "issue", "priority"], default="product")
    parser.add_argument("--max-train-rows", type=int, default=10000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--checkpoint", default=MODEL_NAME,
                        help="Hugging Face model id or local checkpoint directory")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels = load("train", args.task)
    order = np.random.default_rng(42).permutation(len(train_texts))[:args.max_train_rows]
    train_texts = [train_texts[i] for i in order]
    train_labels = [train_labels[i] for i in order]
    classes = sorted(set(train_labels))
    label_to_id = {label: i for i, label in enumerate(classes)}
    encoded_train = [label_to_id[label] for label in train_labels]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=Path(args.checkpoint).exists())
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=len(classes),
        local_files_only=Path(args.checkpoint).exists(), ignore_mismatched_sizes=True,
    )
    training = ComplaintDataset(train_texts, encoded_train, tokenizer, args.max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUT / f"{args.task}-checkpoints"), num_train_epochs=args.epochs,
            per_device_train_batch_size=8, learning_rate=2e-5, weight_decay=0.01,
            save_strategy="no", eval_strategy="no", logging_steps=100,
            report_to="none", seed=42, use_cpu=not torch.backends.mps.is_available(),
        ),
        train_dataset=training,
    )
    trainer.train()
    model_dir = OUT / args.task
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    metric_rows = []
    for split in ("validation", "test"):
        texts, labels = load(split, args.task)
        known = [(text, label_to_id[label]) for text, label in zip(texts, labels) if label in label_to_id]
        ds = ComplaintDataset([x[0] for x in known], [x[1] for x in known], tokenizer, args.max_length)
        pred = trainer.predict(ds)
        y_true = np.asarray([x[1] for x in known])
        y_pred = pred.predictions.argmax(axis=1)
        metric_rows.append([args.task, split, len(labels), len(known), f1_score(y_true, y_pred, average="macro", zero_division=0), accuracy_score(y_true, y_pred)])
        print(f"{args.task} {split}: macro-F1={metric_rows[-1][-2]:.4f}, accuracy={metric_rows[-1][-1]:.4f}, evaluable={len(known)}/{len(labels)}", flush=True)
    with REPORT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0: writer.writerow(["task", "split", "rows", "rows_with_train_label", "macro_f1", "accuracy", "checkpoint", "train_sample", "epochs"])
        writer.writerows([row + [args.checkpoint, len(training), args.epochs] for row in metric_rows])


if __name__ == "__main__":
    main()
