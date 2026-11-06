# Laya-Assisted Literature Screening

A reproducible, resumable pipeline that collects the literature on **underground hydrogen storage (UHS)** from
OpenAlex, screens and classifies it with **Laya** (a non-autoregressive "System-1" decision model) **plus independent
keyword rules, a TF-IDF baseline and human gold labels**, and produces tables, figures, an Excel workbook and an
HTML report suitable as the empirical basis of a *scoping review / bibliometric mapping* paper.

> Scope of this version: papers published **2025-01-01 → today** (`project.start_date` / `end_date: auto`).
> The last year is partial; the report compares *per-month* rates and never raw yearly totals.

---------------------------------------------------------------------------------------------------------------------

## 1. What changed from v2 (and why)

| # | v2 problem (found in the v2 report/code) | v3 fix |
|---|---|---|
| 1 | Classification **never validated**; threshold 0.55 arbitrary; the sample needed Laya scores first | Gold sample is drawn **before** screening, stratified by keyword-rule class (Laya-independent, unbiased). `evaluate` computes design-weighted precision/recall/F1/AUC/ECE with **bootstrap CIs**, picks the threshold for a **recall target (0.95)** and writes `operating_point.json`, which `analyze` applies automatically |
| 2 | Laya used **zero-shot with `noul`** questions. The model card says the base checkpoints are near chance zero-shot, ship **over-confident**, and `noul` can follow its `false:/true:` labels instead of the text (upstream issue #156) | Neutral **A/B `choice`** style by default (`question_style`), `noul` selectable; **Platt calibration** fitted on gold labels; `compare` ranks checkpoint × style on *your* labels; `smoke` tests both styles; fine-tune export |
| 3 | English checkpoint reads only **~320 tokens** of text (512 ctx − 192 head); `max_len: 768` was above the documented context; 258 inputs truncated, silently | Context budget is derived from the loaded checkpoint; **`smart_trim`** keeps title + first sentences + last sentence(s); trimmed records flagged; `smoke` reports the share of your abstracts that exceed the budget; `multilingual` (1024, up to 8192) comparable via `compare` |
| 4 | **30.8 % of records without abstract**; Crossref only | Crossref **and Semantic Scholar** recovery *before* screening; `abstract_source` stored; title-only records are never trusted silently (review-queue priority **E**) |
| 5 | Laya version/checkpoint changes would silently mix results | Every result stores `model_tag = laya version \| checkpoint \| style \| question-hash \| max_len`; `screen` **refuses to mix** (use `--rescreen`, old results backed up inside the DB); version-tolerant API layer (`predict_batch` or `predict`, optional kwargs probed) |
| 6 | "Research gaps: No data" (gap table was restricted to the 3 big settings, so it was empty *by construction*) | All six settings; **absolute gaps** (≤ 3 papers) **and relative gaps** (lift < 0.5 vs. independence) |
| 7 | Over-broad lexicon (e.g. `experiment\|measured` → 33 %; bare `cost`, `stress`, `dissolution`; ambiguous tool names `moose`, `geos`, `ogs`, `eclipse`) | Patterns tightened; new tags (`hydrogen_loss`, OPM Flow, DARTS, STOMP …) |
| 8 | Report/table glitches: empty `decision` column, PRISMA box dropped 5th exclusion reason, overlapping labels, truncated tables, `Mixed` method with 2 papers | Fixed; method classes redesigned (adds `data_driven`, `techno_economic`; drops `mixed`); storage classes add `hard_rock_cavern`, `other_geology` |
| 9 | Hard-coded 2020–2025, no OpenAlex key handling for the Feb-2026 key/credit regime, key stored in `config.yaml` | Window from config; **key only from environment** (or untracked `config.local.yaml`); daily-budget exhaustion stops cleanly and **resumes** |
| 10 | No second rater, no baseline, no CIs | Optional blind second-rater sheet + Cohen's κ + adjudication list; TF-IDF/stack baselines; bootstrap CIs |
| 11 | Review queue not actionable | `human_decision` column → `python run.py ingest-review FILE` → human decisions override the model (`decision_source = human`) |
| 12 | Recall of the *search* was unmeasurable | `validation_inputs/seed_papers.txt` (landmark 2025–2026 reviews pre-filled; add yours) → collection recall + pipeline recall |

**Security:** if you ever pasted `config.yaml` with an OpenAlex key anywhere, **revoke that key** and create a new one.

---------------------------------------------------------------------------------------------------------------------

## 2. Pipeline

```
collect ─► process ─► sample ─► (label in Excel) ─► compare ─► screen ─► evaluate ─► analyze
   │          │          │                             │          │          │           │
OpenAlex   dedup,     blind gold             rank Laya       Laya 2-stage  metrics,    tables, figures,
17+5       Crossref   sheet (rule-             checkpoint ×    (resumable,   calibration,Excel, HTML
queries    + S2       stratified)              question style  version-safe) threshold   report, provenance
           abstracts, rules/tags                                                         
```

1. **collect** – 22 Boolean queries (`config.yaml`), OpenAlex cursor pagination without caps, every query hit stored.
2. **process** – de-duplication (DOI / normalised title, journal version preferred), abstract recovery, eligibility rules
   (errata, retractions, out-of-window, no hydrogen term) and ~50 lexicon tags + keyword-rule second opinion.
3. **sample** – blind labeling sheet (dropdowns). `--second-rater` adds a blind copy of 25 % for a second reviewer.
4. **compare** *(optional but recommended)* – screens **only the gold papers** with every variant in `compare.variants`
   (checkpoint × question style × `max_len`, or your fine-tuned model) and ranks them by precision at the recall target.
5. **screen** – stage 1 (`uhs` question, all eligible) and stage 2 (10 detail questions for raw P ≥ 0.15).
6. **evaluate** – 5-fold cross-validated comparison of *raw Laya, calibrated Laya, keyword rules, TF-IDF logistic baseline,
   stacked model*; bootstrap CIs (resampling inside strata); chooses the deployed scorer/threshold; categorical and topic-flag
   accuracy; inter-rater κ.
7. **analyze** – decisions (model + human overrides), PRISMA counts, ~25 figures, CSVs, Excel, HTML report, provenance bundle.

All state is in one SQLite file (`data/uhs_literature.db`). Every command is resumable and idempotent.

---------------------------------------------------------------------------------------------------------------------

## 3. Quick start (see `GUIDE_FROM_SCRATCH.md` for Windows step-by-step)

```bash
pip install -r requirements.txt
export OPENALEX_API_KEY=...          # Windows: setx OPENALEX_API_KEY "..."   (free key: openalex.org/settings/api)
python run.py doctor                  # environment, key, Laya version
python run.py smoke                   # Laya sanity + which question style works + context-budget check
python run.py collect && python run.py process
python run.py sample --n 400          # label output/validation/gold_labeling_sheet.xlsx (2–3 h)
python run.py compare                 # pick checkpoint/style on your labels -> edit config.yaml
python run.py screen
python run.py evaluate                # validated threshold + metrics; re-generates the report
python run.py analyze                 # at any time; add ingest-review after reviewing the queue
```

Offline self-test of the whole chain (no internet, no weights): `python -m tests.test_offline`.

---------------------------------------------------------------------------------------------------------------------

## 4. Laya: checkpoints, versions and what to change when something new appears

Facts taken from the model card (checked 22 Oct 2026; **re-read it when you upgrade**):

| checkpoint | `screening.checkpoint` | context | text budget | note |
|---|---|---|---|---|
| `convaiinnovations/laya` (root) | `english` | 512 | ≈ 320 tokens | English only |
| `…/laya` subfolder `multilingual` | `multilingual` | 1024 (up to 8192 with `max_len`) | ≈ 768 | 100+ languages, faster |
| `…/laya` subfolder `typed-decisions` | `typed-decisions` | 1024 | ≈ 768 | fine-tuned on four *business* workflows – not on science |
| your fine-tuned repo / local path | `custom` + `model_id` | as trained | – | evaluate with `compare` |
| English for English records, multilingual otherwise | `auto` | – | – | routed by OpenAlex `language` |

* Switching anything = edit `screening:` in `config.yaml`, run `python run.py compare` (cheap: only gold papers), then
  `python run.py screen --rescreen`. The old results are kept in a `laya_backup_*` table.
* `pip install -U laya` is safe: `smoke` probes the answer schema; the version is stored with every result and in the report.
  If a future release changes the output layout, **only `uhs/laya_compat.py` and `uhs/questions.py:canonicalise`** need edits.
* **Fine-tuning** (the model card: accuracy "jumps" with in-domain fine-tuning): `python run.py export-finetune` writes your gold
  labels as JSONL; see `fine_tune/README_FINETUNE.md`. Evaluate the result with `compare` on a **held-out** part of the gold set
  (the export marks a deterministic `test` split).
* Laya's own limits that matter here: base checkpoints near chance zero-shot on its benchmark; ships over-confident
  (calibrate!); `noul` label-dominance on the English checkpoint; `action.act_probability` is unusable (not used here);
  options share a small token budget (we keep ≤ 7 options per question).

---------------------------------------------------------------------------------------------------------------------

## 5. Validation design (what to write in the paper)

* **Gold sample** (n = 400 default): strata = keyword-rule class `strong / likely / unlikely / none` plus a 10 % stratum from the
  *auto-excluded (no hydrogen term)* records (to measure what the rule pre-filter loses). Weight of a paper = stratum size ÷ number
  labeled in that stratum, so population-level precision/recall are unbiased for **any** classifier evaluated on the same labels.
* **Labels**: `human_relevant` (substantially about UHS: 1/0), storage type, method, scale (dropdowns), optional topic flags.
  Judge from title + abstract only (what the model sees).
* **Metrics** (`output/validation/metrics.json`, report §8): weighted AUC, Brier, ECE, precision/recall/F1 with bootstrap 95 % CI,
  estimated number of **missed UHS papers** with CI, categorical accuracy/κ/macro-F1 split by Laya confidence, topic-flag AUC/F1.
* **Operating point**: threshold = largest value whose cross-validated weighted recall ≥ `recall_target`; if unreachable, the
  max-F1 threshold is used and a warning is logged. The scorer defaults to calibrated Laya (`validation.operating_scorer`).
* **Second rater**: κ for relevance and categories; disagreements → `rater_disagreements.csv` → fill `final_relevant`, save as
  `output/validation/adjudication.csv`, re-run `evaluate`.
* Labeled gold papers and ingested review decisions are **human decisions** in the final corpus; they are never used as test data
  for themselves (cross-validation) and are reported separately in the PRISMA flow (human-added / human-removed).

---------------------------------------------------------------------------------------------------------------------

## 6. Outputs

```
output/
  report/UHS_Literature_Report.html      single file, figures embedded (print to PDF)
  excel/UHS_Literature_2025_2026.xlsx    README, Included, Review_Queue, tables…
  tables/*.csv   figures/*.png|svg       every table/figure of the report
  validation/    labeling sheets, metrics.json, operating_point.json, calibration.json, compare_summary.*, smoke_report.*
  provenance/    config (secrets stripped), versions, database SHA-256, operating point, metrics
  logs/          pipeline.log, laya_performance.csv
```

Review queue priorities: **A** possible false negative (rules strong, model negative) · **B** possible false positive ·
**C** rules "likely", model negative · **E** title-only decision · **D** borderline score.

---------------------------------------------------------------------------------------------------------------------

## 7. Configuration cheat-sheet

| Key | Meaning |
|---|---|
| `project.start_date / end_date` | publication window; `end_date: auto` = today |
| `openalex.search_filter` | OpenAlex filter used for Boolean search (`doctor` tests it live) |
| `abstract_enrichment.*` | Crossref / Semantic Scholar recovery |
| `screening.checkpoint / model_id / question_style / max_len / head_max_len / state_mode` | everything Laya-related |
| `screening.stage2_min_prob` | keep low (0.15) so that every included paper gets the detail answers |
| `decision.raw_include_threshold` | used only until `evaluate` writes the validated operating point |
| `validation.*` | sample size/design/allocation, recall target, operating scorer, bootstrap reps, seed file |
| `compare.variants` | what `compare` evaluates |
| `analysis.gap_*` | research-gap definitions |

---------------------------------------------------------------------------------------------------------------------

## 8. Honest limits

* Title/abstract only; one bibliographic database (+ abstract enrichment). For a *systematic* review add Scopus/WoS
  or state the limitation; for pre-registration use OSF/PROSPERO. Call the study a **scoping review / AI-assisted
  bibliometric mapping** unless you do dual human screening and a protocol.
* Categorical answers (storage/method/scale) and topic flags are only as reliable as their validation numbers; lexicon tags mean
  "mentioned in the abstract".
* OpenAlex lags for the most recent months; 2026 is partial; citation counts of 2025–2026 papers are not comparable.
* The offline test proves pipeline logic only. Real accuracy exists only after `evaluate`.
* This project was assembled without access to your machine: Laya was **not executed** here (no weights/network). The API
  layer follows the model card and was tested with fake agents of both call shapes; run `python run.py doctor` and
  `python run.py smoke` first.
