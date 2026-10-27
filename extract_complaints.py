#!/usr/bin/env python3
"""Stream CFPB complaints and persist only complaints with narratives."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from datasets import Dataset, load_dataset


SOURCE_REPO = "Mouwiya/cfpb-consumer-complaints"
UPSTREAM = "CFPB Consumer Complaint Database"
OUTPUT_DIR = Path("data/raw")
SOURCE_NARRATIVE = "consumer_complaint_narrative"
KEEP_COLUMNS = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    SOURCE_NARRATIVE,
    "company",
    "company_response_to_consumer",
    "timely_response",
]
RENAMES = {
    SOURCE_NARRATIVE: "narrative",
    "company_response_to_consumer": "company_response",
    "timely_response": "timely",
}


def main() -> None:
    scanned_by_year: dict[str, int] = defaultdict(int)
    kept_by_year: dict[str, int] = defaultdict(int)
    totals = {"scanned": 0, "kept": 0}
    minimum_date = None
    maximum_date = None
    last_progress = 0

    print(f"Opening {SOURCE_REPO} in streaming mode…", flush=True)
    stream = load_dataset(SOURCE_REPO, split="train", streaming=True)

    def kept_rows():
        nonlocal minimum_date, maximum_date, last_progress
        for row in stream:
            totals["scanned"] += 1
            date_value = row.get("date_received")
            date_text = str(date_value) if date_value is not None else ""
            year = date_text[:4] if len(date_text) >= 4 else "unknown"
            scanned_by_year[year] += 1
            if date_text:
                minimum_date = date_text if minimum_date is None else min(minimum_date, date_text)
                maximum_date = date_text if maximum_date is None else max(maximum_date, date_text)

            narrative = row.get(SOURCE_NARRATIVE)
            if narrative is not None and str(narrative).strip():
                output = {column: row.get(column) for column in KEEP_COLUMNS}
                for old, new in RENAMES.items():
                    output[new] = output.pop(old)
                totals["kept"] += 1
                kept_by_year[year] += 1
                yield output

            if totals["scanned"] - last_progress >= 100_000:
                last_progress = totals["scanned"]
                print(
                    f"Scanned {totals['scanned']:,}; kept {totals['kept']:,}",
                    flush=True,
                )

        print(
            f"Finished scanning {totals['scanned']:,}; kept {totals['kept']:,}",
            flush=True,
        )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(kept_rows, writer_batch_size=10_000)
    dataset.save_to_disk(str(OUTPUT_DIR))

    manifest = {
        "source_repo": SOURCE_REPO,
        "upstream": UPSTREAM,
        "extraction_time_utc": datetime.now(timezone.utc).isoformat(),
        "rows_scanned": totals["scanned"],
        "rows_kept": totals["kept"],
        "column_names": dataset.column_names,
        "min_date_received": minimum_date,
        "max_date_received": maximum_date,
        "rows_scanned_per_year": dict(sorted(scanned_by_year.items())),
        "rows_kept_per_year": dict(sorted(kept_by_year.items())),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved dataset folder: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Extraction failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
