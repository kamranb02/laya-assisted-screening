# From zero to a validated UHS corpus (Windows, step by step)

Time: ~1 h of your attention + ~3 h of labeling + unattended runs (collection: minutes; Laya on CPU: roughly 1–3 h).

## 0. One-time setup
1. Install Python 3.10+ (tick "Add to PATH").
2. Double-click `01_install.bat` (installs the requirements incl. `laya`).
3. Create a **free OpenAlex key** at https://openalex.org/settings/api and store it **in the environment, not in files**:
   `setx OPENALEX_API_KEY "your_key"` → open a **new** terminal. (Optional: `setx S2_API_KEY "..."` for Semantic Scholar.)
4. `00_doctor.bat` — everything should say `[ok]`; it prints the installed Laya version and tests the OpenAlex query filter
   and your daily budget. If the filter name is rejected, change `openalex.search_filter` in `config.yaml`.

## 1. Check Laya on this machine
`02_smoke_test.bat` → `output/validation/smoke_report.md`. It tells you
* whether Laya loads and answers in the expected layout (version-compatibility check),
* which **question style** (`choice_ab` or `noul`) behaves better on a small sanity set with hard negatives,
* how many of your abstracts are longer than the checkpoint's text budget (after step 3 of this guide; run it again then),
* speed and an extrapolated runtime.
If it says FAIL: do not continue; read the report. (The sanity set is tiny — it catches breakage, not real accuracy.)

## 2. Collect and prepare
1. `03_estimate_queries.bat` (optional): one request per query, shows the size of each query.
2. `04_collect.bat` — resumable. If the OpenAlex daily budget runs out it stops cleanly; run again after 00:00 UTC.
3. `05_process.bat` — de-duplication, abstract recovery (Crossref, Semantic Scholar), eligibility rules, tags.
   Check `python run.py status`: "missing abstract" should be clearly lower than before enrichment.
4. Re-run `02_smoke_test.bat` now that the database has abstracts (context-budget check).

## 3. Gold labels (the most valuable 3 hours of the project)
1. `06_make_gold_sample.bat` → `output/validation/gold_labeling_sheet.xlsx`.
2. For each row read title + abstract and fill **human_relevant** (1/0) and, for relevant papers, storage type / method / scale
   (dropdowns). Optional topic columns validate the topic flags. Do not look at any model output while labeling.
3. Recommended: let a colleague label a share blindly — `python run.py sample --second-rater` creates
   `gold_rater2_sheet.xlsx` (25 % of the rows, shuffled).
4. You can save partially and continue later.

## 4. Choose the Laya configuration with evidence
`07_compare_laya_variants.bat` (after ≥ ~100 labels) → `output/validation/compare_summary.md`: precision at the recall target,
AUC and calibration error for every variant in `config.yaml → compare.variants` (english/multilingual/typed-decisions ×
question style). Put the winner into `screening:` (`checkpoint`, `question_style`, `max_len`). New Laya release or a
fine-tuned model? Add it to `compare.variants` and run again.

## 5. Screen
`08_screen_laya.bat`. Resumable (Ctrl+C any time). `python run.py live` (or `11_live_dashboard.bat`) opens a live dashboard.
If you change the configuration afterwards the program refuses to mix results: use `python run.py screen --rescreen`.

## 6. Evaluate → validated corpus
`09_evaluate.bat` → metrics with CIs, calibration, **operating point** (threshold for 95 % recall), and the report is rebuilt
automatically. Read `output/validation/metrics.json` and report §8. If a warning says the stage-2 gate is above the validated
threshold, lower `screening.stage2_min_prob` and re-run `screen`.

## 7. Close the loop on the uncertain papers
Open `output/excel/…xlsx → Review_Queue` (or `output/tables/review_queue.csv`), fill `human_decision` (1/0) for as many
priority-A/B/E rows as you can, save, then
`python run.py ingest-review output/tables/review_queue.csv` — decisions become final (`decision_source = human`).

## 8. Final deliverables
`10_analyze_and_report.bat` → `output/report/UHS_Literature_Report.html`, Excel, figures, `output/provenance/`.
`python run.py recall` checks the landmark papers in `validation_inputs/seed_papers.txt` (add your own).

## Troubleshooting
* **`laya.load()` hangs** → TensorFlow/abseil deadlock; the program already sets `USE_TF=0`. If you start Python yourself, set it too.
* **HTTP 429 / budget** → wait for the reset (00:00 UTC) and re-run `collect`; nothing is lost.
* **`screen` refuses** → configuration changed since the last results; revert or use `--rescreen`.
* **Many `trimmed` texts** → compare `multilingual` (1024 tokens) in step 4.
* **Out of memory** → `screening.batch_size: 4`, or `device: cpu`.
