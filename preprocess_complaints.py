#!/usr/bin/env python3
"""Bounded-memory preprocessing for the CFPB narrative dataset."""

from __future__ import annotations

import csv
import argparse
import hashlib
import random
import re
import sqlite3
import tempfile
import zlib
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk


DATASET_PATH = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
CUTOVER = date(2023, 8, 25)
OLD_PRODUCT = "Credit reporting, credit repair services, or other personal consumer reports"
NEW_PRODUCT = "Credit reporting or other personal consumer reports"
PRODUCT_MAPPING = {OLD_PRODUCT: NEW_PRODUCT, NEW_PRODUCT: NEW_PRODUCT}
MIN_PRODUCT_ROWS = 500
MIN_WORDS = 10
TRAIN_CAP_PER_PRODUCT = 30_000
HOLDOUT_SAMPLE_SIZE = 30_000
RANDOM_SEED = 42
BATCH_SIZE = 5_000
SIMHASH_BITS = 64
SIMHASH_DISTANCE = 3
SIMHASH_MAX_SHINGLES = 96
SIMHASH_BUCKET_LIMIT = 128

FIELDS = [
    "complaint_id", "date_received", "product", "issue", "narrative",
    "company_response", "timely",
]

WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

# CFPB-style masks are converted to typed placeholders before generic X-runs.
MASKED_DATE_RE = re.compile(
    r"(?i)\b(?:\d{1,2}|x{1,2})[/.-](?:\d{1,2}|x{1,2})[/.-](?:\d{2,4}|x{2,4})\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
    r"(?:\d{1,2}|x{1,2}),?\s+(?:\d{2,4}|x{2,4})\b"
)
AMOUNT_MASK_RE = re.compile(
    r"(?i)(?:\$|\bUSD\s*)\s*(?=[0-9x*.,-]{1,24}[x*])[0-9x*]+(?:[,.][0-9x*]+)*(?:\s*dollars?)?"
    r"|\b(?:x{1,3},)?x{3}(?:\.x{2})?\b(?:\s+dollars?)?"
)
X_RUN_RE = re.compile(r"(?i)x{2,}")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\[\]{}]+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SSN_RE = re.compile(r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)")
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"
)


def csv_write(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def row_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def clean_narrative(text: str | None) -> str:
    # Normalize whitespace first so masks and PII patterns see stable boundaries.
    value = WHITESPACE_RE.sub(" ", text or "").strip()

    def redact_date(match: re.Match[str]) -> str:
        return "[REDACTED_DATE]" if re.search(r"(?i)x", match.group()) else match.group()

    value = MASKED_DATE_RE.sub(redact_date, value)
    value = AMOUNT_MASK_RE.sub("[REDACTED_AMOUNT]", value)
    value = X_RUN_RE.sub("[REDACTED]", value)
    value = URL_RE.sub("[URL]", value)
    value = EMAIL_RE.sub("[EMAIL]", value)
    value = SSN_RE.sub("[SSN]", value)
    value = PHONE_RE.sub("[PHONE]", value)
    return WHITESPACE_RE.sub(" ", value).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def simhash64(text: str) -> int:
    tokens = TOKEN_RE.findall(text.casefold())
    if not tokens:
        return 0
    shingles = [" ".join(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))]
    if not shingles:
        shingles = tokens
    # Bound work on the longest narratives with deterministic, evenly-spaced features.
    if len(shingles) > SIMHASH_MAX_SHINGLES:
        step = len(shingles) / SIMHASH_MAX_SHINGLES
        shingles = [shingles[int(i * step)] for i in range(SIMHASH_MAX_SHINGLES)]
    hashes = []
    for shingle in set(shingles):
        encoded = shingle.encode("utf-8", errors="replace")
        high = zlib.crc32(encoded, 0x9E3779B9)
        low = zlib.crc32(encoded, 0x85EBCA6B)
        hashes.append((high << 32) | low)
    if not hashes:
        return 0
    bit_array = np.asarray(hashes, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    votes = np.unpackbits(bit_array, axis=1, bitorder="little").sum(axis=0)
    return sum(1 << bit for bit, ones in enumerate(votes) if int(ones) * 2 >= len(hashes))


def find_root(parent: np.ndarray, item: int) -> int:
    root = item
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[item]) != item:
        next_item = int(parent[item])
        parent[item] = root
        item = next_item
    return root


def split_for_day(day_text: str) -> str:
    day = date.fromisoformat(day_text[:10])
    if day < date(2025, 1, 1):
        return "train"
    if day <= date(2025, 6, 30):
        return "validation"
    return "test"


def write_funnel(path: Path, entries: list[dict[str, object]]) -> None:
    csv_write(
        path,
        ["step", "operation", "rows", "removed_at_step"],
        [[e["step"], e["operation"], e["rows"], e["removed_at_step"]] for e in entries],
    )


def preflight_dataset(ds) -> tuple[dict[str, dict], Counter, set[date]]:
    date_counts: dict[str, Counter] = {OLD_PRODUCT: Counter(), NEW_PRODUCT: Counter()}
    summaries = {
        label: {"count": 0, "min": None, "max": None, "eligible": 0}
        for label in (OLD_PRODUCT, NEW_PRODUCT)
    }
    view = ds.select_columns(["date_received", "product"])
    for batch in view.iter(batch_size=100_000):
        for dt, product in zip(batch["date_received"], batch["product"]):
            if product not in summaries:
                continue
            item = summaries[product]
            item["count"] += 1
            if dt is None:
                continue
            day = dt.date() if hasattr(dt, "date") else dt
            item["min"] = day if item["min"] is None or day < item["min"] else item["min"]
            item["max"] = day if item["max"] is None or day > item["max"] else item["max"]
            date_counts[product][day] += 1
            if day >= date(2018, 1, 1):
                item["eligible"] += 1
    overlap = {
        day for day in date_counts[OLD_PRODUCT]
        if date_counts[NEW_PRODUCT].get(day, 0) > 0
    }
    return summaries, Counter({(label, day): count for label, counts in date_counts.items() for day, count in counts.items()}), overlap


def write_label_mapping(path: Path, summaries: dict[str, dict], daily: Counter) -> None:
    rows = []
    for source, canonical in PRODUCT_MAPPING.items():
        summary = summaries[source]
        rows.append([
            "mapping", source, canonical, "", summary["count"],
            summary["min"], summary["max"], CUTOVER, summary["eligible"],
        ])
    for daynum in range(20, 32):
        day = date(2023, 8, daynum)
        for source, canonical in PRODUCT_MAPPING.items():
            rows.append([
                "daily_cutover_count", source, canonical, day,
                daily.get((source, day), 0), summaries[source]["min"],
                summaries[source]["max"], CUTOVER, summaries[source]["eligible"],
            ])
    csv_write(
        path,
        ["record_type", "source_label", "canonical_label", "date_received", "row_count",
         "observed_min_date", "observed_max_date", "cutover_date", "eligible_rows_after_step_1"],
        rows,
    )


def new_database(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-131072")  # about 128 MiB
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript(
        """
        CREATE TABLE stage(
            row_id INTEGER PRIMARY KEY,
            complaint_id TEXT,
            date_received TEXT NOT NULL,
            product TEXT,
            issue TEXT,
            narrative BLOB NOT NULL,
            company_response TEXT,
            timely TEXT,
            year INTEGER NOT NULL,
            word_count INTEGER NOT NULL
        );
        CREATE TABLE exact_hashes(hash TEXT PRIMARY KEY, keep_id INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE exact_keep(idx INTEGER PRIMARY KEY, row_id INTEGER UNIQUE NOT NULL, exact_hash TEXT NOT NULL);
        CREATE TABLE lsh(band INTEGER NOT NULL, band_key INTEGER NOT NULL, idx INTEGER NOT NULL,
                         PRIMARY KEY(band, band_key, idx)) WITHOUT ROWID;
        CREATE TABLE near_keep(row_id INTEGER PRIMARY KEY);
        CREATE TABLE split_members(
            row_id INTEGER PRIMARY KEY,
            split TEXT NOT NULL,
            product TEXT NOT NULL,
            complaint_id TEXT,
            text_hash TEXT NOT NULL,
            priority INTEGER NOT NULL
        );
        """
    )
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing train.parquet, validation.parquet, and test.parquet outputs",
    )
    args = parser.parse_args()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    ds = load_from_disk(str(DATASET_PATH))
    print(f"load_from_disk confirmed: {DATASET_PATH} ({type(ds).__name__}, {len(ds):,} rows)", flush=True)
    expected = {"complaint_id", "date_received", "product", "issue", "narrative", "company_response", "timely"}
    missing = expected.difference(ds.column_names)
    if missing:
        raise ValueError(f"Required dataset columns are missing: {sorted(missing)}")

    summaries, daily_counts, overlap_days = preflight_dataset(ds)
    print("Daily label counts, 2023-08-20 through 2023-08-31", flush=True)
    for daynum in range(20, 32):
        day = date(2023, 8, daynum)
        print(
            f"{day}: old={daily_counts.get((OLD_PRODUCT, day), 0):,}; "
            f"new={daily_counts.get((NEW_PRODUCT, day), 0):,}",
            flush=True,
        )
    for label in (OLD_PRODUCT, NEW_PRODUCT):
        item = summaries[label]
        print(f"Label range: {label}: {item['min']} to {item['max']} ({item['count']:,} rows)", flush=True)
    if overlap_days != {CUTOVER}:
        raise RuntimeError(
            "Credit-reporting labels have sustained or unexpected overlap; "
            f"overlap dates={sorted(overlap_days)}. No processed outputs were created."
        )
    print(f"Confirmed single-day label overlap at {CUTOVER}; proceeding with mapping.", flush=True)

    expected_outputs = {"train.parquet", "validation.parquet", "test.parquet"}
    if PROCESSED_DIR.exists() and any(PROCESSED_DIR.iterdir()):
        existing = {p.name for p in PROCESSED_DIR.iterdir() if p.is_file()}
        if not args.overwrite or existing != expected_outputs:
            raise FileExistsError(
                f"{PROCESSED_DIR} is not empty; use --overwrite only when it contains exactly {sorted(expected_outputs)}."
            )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_label_mapping(REPORTS_DIR / "label_mapping.csv", summaries, daily_counts)

    funnel: list[dict[str, object]] = []

    def log(step: str, operation: str, rows: int, previous: int | None = None) -> None:
        removed = 0 if previous is None else previous - rows
        entry = {"step": step, "operation": operation, "rows": rows, "removed_at_step": removed}
        funnel.append(entry)
        write_funnel(REPORTS_DIR / "cleaning_funnel.csv", funnel)
        print(f"{step}: {operation}: {rows:,} rows (removed {removed:,})", flush=True)

    log("input", "load_from_disk data/raw", len(ds))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cfpb_preprocess_", dir="/private/tmp") as temp_root:
        conn = new_database(Path(temp_root) / "stage.sqlite")
        try:
            stage_insert = """INSERT INTO stage
                (complaint_id,date_received,product,issue,narrative,company_response,timely,year,word_count)
                VALUES(?,?,?,?,?,?,?,?,?)"""
            batch_rows = []
            input_rows = 0
            valid_date_rows = 0
            step1_rows = 0
            view = ds.select_columns([
                "complaint_id", "date_received", "product", "issue", "narrative",
                "company_response", "timely",
            ])
            for batch in view.iter(batch_size=BATCH_SIZE):
                input_rows += len(batch["date_received"])
                for complaint_id, dt, product, issue, narrative, response, timely in zip(
                    batch["complaint_id"], batch["date_received"], batch["product"],
                    batch["issue"], batch["narrative"], batch["company_response"], batch["timely"],
                ):
                    if dt is None:
                        continue
                    valid_date_rows += 1
                    day = dt.date() if hasattr(dt, "date") else dt
                    if day < date(2018, 1, 1):
                        continue
                    step1_rows += 1
                    normalized = clean_narrative(narrative)
                    stored_text = zlib.compress(normalized.encode("utf-8"), level=1)
                    batch_rows.append((
                        complaint_id, to_iso(dt), product, issue, stored_text,
                        response, timely, day.year, word_count(normalized),
                    ))
                if len(batch_rows) >= 20_000:
                    conn.executemany(stage_insert, batch_rows)
                    conn.commit()
                    batch_rows.clear()
                    print(f"Filtered and normalized rows staged: {step1_rows:,}", flush=True)
            if batch_rows:
                conn.executemany(stage_insert, batch_rows)
                conn.commit()
            if input_rows != len(ds):
                raise AssertionError(f"Read {input_rows:,} rows but dataset reports {len(ds):,}.")
            log("1a", "drop rows without valid date_received", valid_date_rows, len(ds))
            log("1b", "keep date_received >= 2018-01-01", step1_rows, valid_date_rows)
            log("2", "normalize whitespace, standardize masks, scrub PII", row_count(conn, "stage"))

            conn.execute("UPDATE stage SET product=? WHERE product IN (?,?)", (NEW_PRODUCT, OLD_PRODUCT, NEW_PRODUCT))
            mapped_counts = Counter()
            for product, count in conn.execute("SELECT product,COUNT(*) FROM stage GROUP BY product"):
                mapped_counts[product or "[MISSING]"] = count
            log("3a", "map both credit-reporting labels to canonical label", row_count(conn, "stage"))
            rare_products = sorted(
                (product, count) for product, count in mapped_counts.items()
                if count < MIN_PRODUCT_ROWS and product != NEW_PRODUCT
            )
            csv_write(
                REPORTS_DIR / "rare_products_dropped.csv",
                ["product", "rows_before_drop", "threshold", "action"],
                [[p, n, MIN_PRODUCT_ROWS, "dropped"] for p, n in rare_products],
            )
            if rare_products:
                for product, _count in rare_products:
                    if product == "[MISSING]":
                        conn.execute("DELETE FROM stage WHERE product IS NULL OR trim(product)=''")
                    else:
                        conn.execute("DELETE FROM stage WHERE product=?", (product,))
                conn.commit()
            after_rare = row_count(conn, "stage")
            log("3b", "drop products with fewer than 500 rows", after_rare, step1_rows)
            print("Rare products dropped:", rare_products or "none", flush=True)

            before_words = after_rare
            conn.execute("DELETE FROM stage WHERE word_count < ?", (MIN_WORDS,))
            conn.commit()
            after_words = row_count(conn, "stage")
            log("4", "drop narratives under 10 words", after_words, before_words)

            conn.execute("CREATE INDEX stage_order ON stage(date_received,row_id)")
            exact_removed = Counter()
            exact_insert = "INSERT INTO exact_keep(idx,row_id,exact_hash) VALUES(?,?,?)"
            exact_seen_insert = "INSERT OR IGNORE INTO exact_hashes(hash,keep_id) VALUES(?,?)"
            exact_idx = 0
            cursor = conn.execute("SELECT row_id,date_received,product,narrative,year FROM stage ORDER BY date_received,row_id")
            for row_id, received, product, compressed, year in cursor:
                text_value = zlib.decompress(compressed).decode("utf-8")
                digest = hashlib.sha256(text_value.casefold().encode("utf-8")).hexdigest()
                result = conn.execute(exact_seen_insert, (digest, row_id))
                if result.rowcount:
                    exact_idx += 1
                    conn.execute(exact_insert, (exact_idx, row_id, digest))
                else:
                    exact_removed[(year, product or "[MISSING]")] += 1
                if exact_idx and exact_idx % 50_000 == 0:
                    conn.commit()
                    print(f"Exact-dedup scan: {exact_idx:,} unique rows seen", flush=True)
            conn.commit()
            after_exact = row_count(conn, "exact_keep")
            log("5a", "drop exact case-insensitive narrative duplicates; keep earliest", after_exact, after_words)

            exact_count = after_exact
            parent = np.arange(exact_count, dtype=np.int32)
            representative = np.arange(exact_count, dtype=np.int32)
            kept = np.zeros(exact_count, dtype=np.bool_)
            signatures = np.zeros(exact_count, dtype=np.uint64)
            year_codes = np.zeros(exact_count, dtype=np.int16)
            product_names = sorted(mapped_counts)
            product_code = {name: ix for ix, name in enumerate(product_names)}
            product_codes = np.zeros(exact_count, dtype=np.int16)
            row_ids = np.zeros(exact_count, dtype=np.int64)
            near_removed = Counter()
            lsh_buffer = []
            near_select = conn.execute(
                """SELECT e.idx,s.row_id,s.date_received,s.product,s.narrative,s.year
                   FROM exact_keep e JOIN stage s ON s.row_id=e.row_id ORDER BY e.idx"""
            )
            for idx, (exact_idx, row_id, received, product, compressed, year) in enumerate(near_select):
                text_value = zlib.decompress(compressed).decode("utf-8")
                sig = simhash64(text_value)
                signatures[idx] = sig
                row_ids[idx] = row_id
                year_codes[idx] = year
                product_codes[idx] = product_code.get(product or "[MISSING]", 0)
                candidate_roots = set()
                for band in range(4):
                    band_key = (sig >> (16 * band)) & 0xFFFF
                    candidate_ids = conn.execute(
                        "SELECT idx FROM lsh WHERE band=? AND band_key=? ORDER BY idx LIMIT ?",
                        (band, band_key, SIMHASH_BUCKET_LIMIT),
                    ).fetchall()
                    for (candidate_idx,) in candidate_ids:
                        candidate_sig = int(signatures[candidate_idx])
                        if (sig ^ candidate_sig).bit_count() <= SIMHASH_DISTANCE:
                            candidate_roots.add(find_root(parent, int(representative[candidate_idx])))
                if candidate_roots:
                    root = min(candidate_roots)
                    for other in sorted(candidate_roots):
                        other = find_root(parent, other)
                        if other != root:
                            parent[other] = root
                            if kept[other]:
                                kept[other] = False
                                other_product = product_names[int(product_codes[other])]
                                near_removed[(int(year_codes[other]), other_product)] += 1
                    parent[idx] = root
                    representative[idx] = root
                    near_removed[(year, product or "[MISSING]")] += 1
                else:
                    parent[idx] = idx
                    representative[idx] = idx
                    kept[idx] = True
                for band in range(4):
                    lsh_buffer.append((band, (sig >> (16 * band)) & 0xFFFF, idx))
                if len(lsh_buffer) >= 20_000:
                    conn.executemany("INSERT INTO lsh(band,band_key,idx) VALUES(?,?,?)", lsh_buffer)
                    conn.commit()
                    lsh_buffer.clear()
                if (idx + 1) % 50_000 == 0:
                    print(f"SimHash near-dedup scan: {idx + 1:,}/{exact_count:,}", flush=True)
            if lsh_buffer:
                conn.executemany("INSERT INTO lsh(band,band_key,idx) VALUES(?,?,?)", lsh_buffer)
                conn.commit()
            for idx, retain in enumerate(kept):
                if retain:
                    conn.execute("INSERT INTO near_keep(row_id) VALUES(?)", (int(row_ids[idx]),))
            conn.commit()
            after_near = row_count(conn, "near_keep")
            log("5b", "SimHash 3-gram clusters (64-bit, Hamming distance <= 3); keep earliest representative", after_near, after_exact)

            conn.execute(
                """INSERT INTO split_members(row_id,split,product,complaint_id,text_hash,priority)
                   SELECT s.row_id,
                     CASE WHEN substr(s.date_received,1,10) <= '2024-12-31' THEN 'train'
                          WHEN substr(s.date_received,1,10) <= '2025-06-30' THEN 'validation'
                          ELSE 'test' END,
                     COALESCE(s.product,'[MISSING]'),s.complaint_id,e.exact_hash,
                     CASE WHEN lower(COALESCE(s.timely,''))='no'
                               OR instr(lower(COALESCE(s.company_response,'')),'monetary relief')>0
                          THEN 1 ELSE 0 END
                   FROM near_keep n JOIN stage s ON s.row_id=n.row_id
                   JOIN exact_keep e ON e.row_id=s.row_id"""
            )
            conn.commit()
            id_overlap = conn.execute(
                "SELECT complaint_id FROM split_members WHERE complaint_id IS NOT NULL GROUP BY complaint_id HAVING COUNT(DISTINCT split)>1 LIMIT 1"
            ).fetchone()
            hash_overlap = conn.execute(
                "SELECT text_hash FROM split_members GROUP BY text_hash HAVING COUNT(DISTINCT split)>1 LIMIT 1"
            ).fetchone()
            if id_overlap or hash_overlap:
                raise AssertionError(f"Cross-split leakage: complaint_id={id_overlap}, text_hash={hash_overlap}")
            split_counts = dict(conn.execute("SELECT split,COUNT(*) FROM split_members GROUP BY split"))
            train_products = {r[0] for r in conn.execute("SELECT DISTINCT product FROM split_members WHERE split='train'")}
            holdout_products = {r[0] for r in conn.execute("SELECT DISTINCT product FROM split_members WHERE split IN ('validation','test')")}
            missing_train_products = sorted(holdout_products - train_products)
            csv_write(
                REPORTS_DIR / "products_missing_from_train.csv",
                ["product", "present_in_validation_or_test", "present_in_train"],
                [[p, True, False] for p in missing_train_products],
            )
            log("6a", "date split before subsampling: train", split_counts.get("train", 0))
            log("6b", "date split before subsampling: validation", split_counts.get("validation", 0))
            log("6c", "date split before subsampling: test", split_counts.get("test", 0))
            print("Products in validation/test but missing from train:", missing_train_products or "none", flush=True)
            print("Cross-split complaint_id and case-insensitive text-hash assertions passed.", flush=True)

            # Reservoirs are bounded by the requested final sample sizes.
            rng = random.Random(RANDOM_SEED)
            train_seen = Counter()
            train_reservoir: dict[str, list] = defaultdict(list)
            holdout_seen = Counter()
            holdout_reservoir: dict[str, list] = {"validation": [], "test": []}
            select = conn.execute(
                """SELECT sm.split,sm.product,sm.priority,s.complaint_id,s.date_received,s.issue,
                          s.narrative,s.company_response,s.timely
                   FROM split_members sm JOIN stage s ON s.row_id=sm.row_id
                   ORDER BY sm.split,sm.row_id"""
            )
            for split, product, priority, complaint_id, received, issue, compressed, response, timely in select:
                narrative = zlib.decompress(compressed).decode("utf-8")
                record = (
                    complaint_id, datetime.fromisoformat(received), product, issue,
                    narrative, response, timely, int(priority),
                )
                if split == "train":
                    train_seen[product] += 1
                    reservoir = train_reservoir[product]
                    if len(reservoir) < TRAIN_CAP_PER_PRODUCT:
                        reservoir.append(record)
                    else:
                        pick = rng.randrange(train_seen[product])
                        if pick < TRAIN_CAP_PER_PRODUCT:
                            reservoir[pick] = record
                else:
                    holdout_seen[split] += 1
                    reservoir = holdout_reservoir[split]
                    if len(reservoir) < HOLDOUT_SAMPLE_SIZE:
                        reservoir.append(record)
                    else:
                        pick = rng.randrange(holdout_seen[split])
                        if pick < HOLDOUT_SAMPLE_SIZE:
                            reservoir[pick] = record

            output_rows = {
                "train": [r for p in sorted(train_reservoir) for r in train_reservoir[p]],
                "validation": holdout_reservoir["validation"],
                "test": holdout_reservoir["test"],
            }
            log("7a", "train cap at 30,000 rows per product", sum(map(len, train_reservoir.values())), split_counts.get("train", 0))
            log("7b", "uniform random validation sample (max 30,000)", len(output_rows["validation"]), split_counts.get("validation", 0))
            log("7c", "uniform random test sample (max 30,000)", len(output_rows["test"]), split_counts.get("test", 0))

            # Verify the sampled artifacts as well before writing them.
            sampled_ids: dict[str, str] = {}
            sampled_hashes: dict[str, str] = {}
            rates = Counter()
            for split, records in output_rows.items():
                for record in records:
                    complaint_id, _, product, _, narrative, _, _, priority = record
                    digest = hashlib.sha256(narrative.casefold().encode("utf-8")).hexdigest()
                    if complaint_id is not None:
                        previous = sampled_ids.setdefault(str(complaint_id), split)
                        if previous != split:
                            raise AssertionError(f"Sampled complaint_id leaked between {previous} and {split}.")
                    previous = sampled_hashes.setdefault(digest, split)
                    if previous != split:
                        raise AssertionError(f"Sampled text hash leaked between {previous} and {split}.")
                    rates[(split, product, "rows")] += 1
                    rates[(split, product, "positive")] += priority

            out_dir = Path(tempfile.mkdtemp(prefix="outputs_", dir=str(PROCESSED_DIR)))
            try:
                schema = pa.schema([
                    pa.field("complaint_id", pa.string()),
                    pa.field("date_received", pa.timestamp("us", tz="UTC")),
                    pa.field("product", pa.string()),
                    pa.field("issue", pa.string()),
                    pa.field("narrative", pa.string()),
                    pa.field("company_response", pa.string()),
                    pa.field("timely", pa.string()),
                ])
                for split, records in output_rows.items():
                    arrays = {
                        name: [record[index] for record in records]
                        for index, name in enumerate(FIELDS)
                    }
                    table = pa.Table.from_pydict(arrays, schema=schema)
                    if table.num_rows != len(records):
                        raise AssertionError(f"Serialized {split} row count does not match sampled count.")
                    pq.write_table(table, out_dir / f"{split}.parquet", compression="zstd")
                for split in ("train", "validation", "test"):
                    (out_dir / f"{split}.parquet").replace(PROCESSED_DIR / f"{split}.parquet")
            finally:
                if out_dir.exists():
                    for path in out_dir.iterdir():
                        path.unlink()
                    out_dir.rmdir()

            rate_rows = []
            for split, product, _ in sorted(rates):
                if _ != "rows":
                    continue
                n = rates[(split, product, "rows")]
                positives = rates[(split, product, "positive")]
                rate_rows.append([split, product, n, positives, positives / n if n else 0])
            csv_write(
                REPORTS_DIR / "priority_positive_rates.csv",
                ["split", "product", "rows", "positive_rows", "positive_rate"],
                rate_rows,
            )

            removed_rows = []
            for (year, product), count in sorted(exact_removed.items()):
                removed_rows.append(["exact", year, product, count])
            for (year, product), count in sorted(near_removed.items()):
                removed_rows.append(["near", year, product, count])
            csv_write(
                REPORTS_DIR / "dedup_removed_by_year_product.csv",
                ["dedup_stage", "year", "product", "rows_removed"],
                removed_rows,
            )
            csv_write(
                REPORTS_DIR / "dedup_method.csv",
                ["stage", "method", "parameters", "caveat"],
                [
                    ["exact", "SHA-256 of casefolded, whitespace-normalized narrative", "keep earliest date_received; row order breaks ties", "casefold is used for hashing only; saved narrative casing is preserved"],
                    ["near", "64-bit SimHash over word 3-grams", f"Hamming distance <= {SIMHASH_DISTANCE}; four 16-bit bands; at most {SIMHASH_BUCKET_LIMIT} prior candidates per band; at most {SIMHASH_MAX_SHINGLES} evenly spaced shingles per narrative", "candidate buckets are capped for bounded work; near-duplicate clustering is approximate"],
                ],
            )
            print("Priority positive rates by split and product:", flush=True)
            for row in rate_rows:
                print(f"{row[0]} | {row[1]}: {row[3]:,}/{row[2]:,} = {row[4]:.4%}", flush=True)
            print("Saved splits:", {k: len(v) for k, v in output_rows.items()}, flush=True)
            print("Output columns:", FIELDS, flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
