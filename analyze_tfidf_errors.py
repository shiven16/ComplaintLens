#!/usr/bin/env python3
"""Write validation-only TF-IDF confusion summaries and representative errors."""
import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix

DATA = Path("data/processed")
MODEL = Path("models/tfidf")
REPORTS = Path("reports")


def main():
    vectorizer = joblib.load(MODEL / "vectorizer.joblib")
    models = {task: joblib.load(MODEL / f"{task}_logistic_regression.joblib") for task in ("product", "issue", "priority")}
    data = pq.read_table(DATA / "validation.parquet", columns=["complaint_id", "narrative", "product", "issue", "timely", "company_response"]).to_pydict()
    texts = [x or "" for x in data["narrative"]]
    x = vectorizer.transform(texts)
    truths = {"product": data["product"], "issue": data["issue"], "priority": [int(t == "No" or "monetary relief" in (r or "").casefold()) for t, r in zip(data["timely"], data["company_response"])]}
    confusion_rows, error_rows = [], []
    for task, model in models.items():
        truth = truths[task]
        pred = model.predict(x)
        labels = model.classes_.tolist()
        mat = confusion_matrix(truth, pred, labels=labels)
        counts = Counter()
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                if i != j and mat[i, j]: counts[(actual, predicted)] += int(mat[i, j])
        for (actual, predicted), n in counts.most_common(30):
            confusion_rows.append([task, actual, predicted, n])
        wrong = np.flatnonzero(np.asarray(truth, dtype=object) != pred.astype(object))
        for i in wrong[:50]:
            error_rows.append([task, data["complaint_id"][i], truth[i], pred[i], texts[i][:1000]])

    with (REPORTS / "tfidf_validation_top_confusions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task", "actual", "predicted", "count"]); w.writerows(confusion_rows)
    with (REPORTS / "tfidf_validation_error_examples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task", "complaint_id", "actual", "predicted", "narrative_excerpt"]); w.writerows(error_rows)
    print(f"Saved top confusions and {len(error_rows):,} representative errors from validation only.")


if __name__ == "__main__":
    main()
