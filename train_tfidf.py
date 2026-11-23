#!/usr/bin/env python3
"""Train TF-IDF logistic-regression baselines on the fixed CFPB splits."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score

DATA = Path("data/processed")
REPORTS = Path("reports")
MODEL_DIR = Path("models/tfidf")
SPLITS = ("train", "validation", "test")
TASKS = ("product", "issue", "priority")


def load_split(name: str) -> dict[str, list]:
    table = pq.read_table(DATA / f"{name}.parquet", columns=["narrative", "product", "issue", "timely", "company_response"])
    data = table.to_pydict()
    data["narrative"] = [text or "" for text in data["narrative"]]
    data["priority"] = [
        int(timely == "No" or "monetary relief" in (response or "").casefold())
        for timely, response in zip(data["timely"], data["company_response"])
    ]
    return data


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {split: load_split(split) for split in SPLITS}
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=2, max_features=300_000, sublinear_tf=True, dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(datasets["train"]["narrative"])
    matrices = {"train": x_train}
    for split in ("validation", "test"):
        matrices[split] = vectorizer.transform(datasets[split]["narrative"])
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib", compress=3)

    metric_rows = []
    class_rows = []
    for task in TASKS:
        y_train = np.asarray(datasets["train"][task])
        # SGD with log loss is logistic regression and scales better than batch
        # LBFGS on this large, high-dimensional sparse text matrix.
        model = SGDClassifier(
            loss="log_loss", max_iter=12, tol=1e-3, random_state=42,
            average=True, class_weight=None, n_jobs=-1,
        )
        model.fit(x_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{task}_logistic_regression.joblib", compress=3)
        for split in ("validation", "test"):
            y_true = np.asarray(datasets[split][task])
            y_pred = model.predict(matrices[split])
            metric_rows.append([
                task, split, len(y_true), f1_score(y_true, y_pred, average="macro", zero_division=0),
                accuracy_score(y_true, y_pred),
                top_k_accuracy_score(y_true, model.predict_proba(matrices[split]),
                                     k=3, labels=model.classes_) if task == "issue" else "",
            ])
            report = classification_report(y_true, y_pred, labels=model.classes_,
                                           output_dict=True, zero_division=0)
            for label in model.classes_:
                item = report[str(label)]
                class_rows.append([task, split, label, int(item["support"]), item["precision"], item["recall"], item["f1-score"]])
        print(f"Finished {task}: {len(model.classes_)} classes", flush=True)

    with (REPORTS / "tfidf_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["task", "split", "rows", "macro_f1", "accuracy", "top3_accuracy"]); writer.writerows(metric_rows)
    with (REPORTS / "tfidf_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["task", "split", "class", "support", "precision", "recall", "f1"]); writer.writerows(class_rows)
    metadata = {
        "model": "TfidfVectorizer + SGDClassifier(log_loss; averaged logistic regression)",
        "training_split": "train.parquet only",
        "evaluation_splits": ["validation.parquet", "test.parquet"],
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_features": 300000, "sublinear_tf": True},
        "tasks": list(TASKS), "random_state": 42,
        "priority_definition": "timely == No OR company_response contains monetary relief (case-insensitive)",
    }
    (MODEL_DIR / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {REPORTS / 'tfidf_metrics.csv'}; per-class results: {REPORTS / 'tfidf_per_class.csv'}")


if __name__ == "__main__":
    main()
