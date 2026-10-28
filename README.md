# ComplaintLens

ComplaintLens is a university NLP project in the industry track on automated routing and priority detection for financial consumer complaints.

## Problem

Financial institutions receive large volumes of free-text complaints that need routing to the right team and timely attention. Manual triage can be slow and inconsistent. ComplaintLens explores NLP models that classify a complaint narrative by product and issue and flag complaints for priority review.

The intended users are complaint triage agents, compliance officers, and operations managers. The business objective is to reduce time to correct routing and surface potentially high-risk complaints earlier. Product and issue are the consumer’s own selected categories; this project uses them as proxies for routing labels, not as independently verified ground truth. The candidate priority flag is also a proxy, based on CFPB response fields.

## Dataset

The raw data is a third-party Hugging Face copy, [Mouwiya/cfpb-consumer-complaints](https://huggingface.co/datasets/Mouwiya/cfpb-consumer-complaints), with the CFPB Consumer Complaint Database as its upstream source. The live database no longer publishes narratives (Release 24, September 2026; CFPB announced the policy on August 14); previously published narratives are available through the [CFPB FOIA reading-room archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). The project uses the third-party snapshot so the narrative field is available alongside the complaint records. This is not a continuously refreshable copy of the live database.

The manifest records **16,558,052** scanned rows and **3,815,251** rows with non-empty narratives. Dates range from **2011-12-01** to **2026-07-06**; **45,040** rows have no valid date. The extracted fields are complaint_id, date_received, product, sub_product, issue, sub_issue, company, narrative, company_response, and timely.

Narrative rate by year from the manifest: 2011 **0.00%**, 2012 **0.00%**, 2013 **0.00%**, 2014 **0.00%**, 2015 **32.52%**, 2016 **40.65%**, 2017 **47.50%**, 2018 **46.04%**, 2019 **45.03%**, 2020 **39.24%**, 2021 **41.05%**, 2022 **42.15%**, 2023 **37.78%**, 2024 **29.78%**, 2025 **22.45%**, 2026 **2.20%**, unknown date **37.13%**.

Narratives are optional and published only with consumer consent, so this is a selected, one-sided, non-representative sample—not a representative sample of all consumers or financial experiences ([CFPB publication criteria](https://files.consumerfinance.gov/f/documents/cfpb_narrative-scrubbing-standard_2023-05.pdf); [database limitations](https://www.consumerfinance.gov/data-research/consumer-complaints/)). Narrative availability falls sharply in 2026. The product mix is dominated by credit reporting; the three largest company labels are Equifax, TransUnion, and Experian. The Hugging Face dataset card reports its license as other; CFPB states that previously published narratives are in the public domain for FOIA purposes. Check the [dataset card](https://huggingface.co/datasets/Mouwiya/cfpb-consumer-complaints) and CFPB archive for the applicable terms and provenance.

## Exploratory analysis

The findings below come from the existing tables and figures in reports/; narrative-text results use a reproducible 200,000-row sample unless otherwise stated.

- Product labels shift around 2017: legacy “Credit reporting” coexists with the longer credit-reporting label, which is later renamed. The long and new labels meet on 2023-08-25; see reports/product_mix_by_year.svg and reports/label_mapping.csv.
- The raw product distribution has 21 labels; its largest category is 43.8% and its smallest has 16 rows (about 104,434:1). For issues, the largest category is 31.6%, with a largest-to-smallest nonzero count ratio above 1.2 million.
- Sample narrative length is 116 words at the median, 507 at p95, and 1,002 at p99. The sample includes 145 narratives under five words and 200 above its p99.9 threshold. See reports/narrative_length_distribution.svg.
- Exact duplicates participate in 46,537 of 200,000 sampled rows (23.3%), especially credit reporting and 2024–25. Approximate SimHash found 596,289 near-duplicate pairs involving 31,436 unique sample rows; these concentrate in credit reporting and 2024–25. Candidate bucket caps make this estimate approximate. See reports/duplicate_signals_by_year.svg.
- CFPB masks are common: 138,899 sample rows match X-runs and 134,747 match amount masks. Residual regex matches were rare (17 email, 5 phone, 2 SSN-like, and 3 card-like rows); these are pattern matches, not confirmed PII. A 20,000-row language-ID subset classified 99.36% of eligible narratives as English. Both are sample-based estimates.
- The top three company labels account for 61.0% of complaints overall; their combined share reaches 74.9% in 2024 and 71.9% in 2025. Timely responses are about 98.9% overall, varying by year and product. See reports/top_company_concentration.svg.

## Preprocessing

The funnel below is recorded in reports/cleaning_funnel.csv. Counts after date splits are before the final sampling step.

| Step | Operation | Rows |
|---|---|---:|
| Input | Narratives in data/raw | 3,815,251 |
| 1a | Drop rows without a valid received date | 3,770,211 |
| 1b | Keep dates from 2018-01-01 onward | 3,522,411 |
| 2 | Normalize whitespace, standardize CFPB masks, scrub emails, phones, SSN-like strings, and URLs | 3,522,411 |
| 3 | Merge credit-reporting labels; no products under the 500-row threshold were dropped | 3,522,411 |
| 4 | Drop narratives under 10 words | 3,496,816 |
| 5a | Exact case-insensitive narrative deduplication; keep earliest | 2,176,228 |
| 5b | Approximate SimHash 3-gram clustering; keep earliest representative | 2,064,855 |
| 6 | Date split: train 2018–2024; validation 2025 H1; test 2025 H2 onward | 1,433,929 / 306,314 / 324,612 |
| 7 | Subsample after split: train capped at 30,000 per product; validation and test uniformly sampled | 298,397 / 30,000 / 30,000 |

The two credit-reporting labels are mapped to “Credit reporting or other personal consumer reports”; the single-day cutover and daily counts are in reports/label_mapping.csv. SimHash parameters and limitations are in reports/dedup_method.csv. Leakage checks assert that no complaint ID or normalized text hash occurs in multiple splits. Products absent from train but present in validation or test are listed in reports/products_missing_from_train.csv.

This produces a test set without duplicate complaints. Train’s product mix is capped and is not the natural distribution, while validation and test are uniform samples of their respective time periods. The priority target is a proxy: (timely == "No") OR company_response contains "monetary relief"; it is not an adjudicated risk label.

## Reproduction

Run from the repository root. data/ is Git-ignored and requires about **5 GB** of disk space.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python extract_complaints.py

Extraction streams the Hugging Face dataset and saves data/raw/ plus data/raw/manifest.json. There is no raw-EDA runner in this repository; the EDA outputs in reports/ are existing artifacts and are not regenerated by a checked-in script.

    python preprocess_complaints.py --overwrite
    python analyze_processed.py

Preprocessing writes train, validation, and test Parquet files under data/processed/ and cleaning metadata under reports/. The processed-split analysis writes summary tables and SVG figures under reports/. Requirements are pinned in requirements.txt.

## Plan

- **M3:** TF-IDF with logistic regression, then fine-tuned DistilBERT; maintain an experiment log. Optional add-ons: priority flag prediction and similar-complaint retrieval.
- **M4:** Evaluate macro-F1, per-class results, top-3 issue accuracy, and error analysis; test robustness to masked company names, noisy text, and later time periods.
- **M5:** Build a FastAPI service and small Streamlit demo; measure latency and add monitoring and failure handling.
- **M6:** Complete the report, demo, and viva.
