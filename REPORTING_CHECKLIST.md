# Reporting checklist for an AI-assisted review (map to this project)

Cross-check against the official PRISMA 2020, PRISMA-S and PRISMA-trAIce (JMIR AI 2025;4:e80247) items before submission — this
table lists what an AI-assisted review should disclose and **where the project stores it**.

| Information | Where |
|---|---|
| Review type, protocol/registration, eligibility criteria | write in paper; criteria = question text in `uhs/questions.py` (`Q_UHS`) |
| Databases, dates of search, full Boolean queries, filters, limits | report appendix; `config.yaml → queries`; `search_log` table; window in report header |
| De-duplication method and numbers | PRISMA figure; `process.dedup` |
| Abstract recovery sources and share | report data-quality table (`abstract_source_*`) |
| AI tool name, version, checkpoint, revision, hardware | report "Model and settings used"; `output/provenance/provenance.json`; `meta` table |
| Exact prompts / questions and answer options | `uhs/questions.py` (hash `question_hash` in every `model_tag`) |
| Input given to the model (title+abstract, trimming rule) | `screening.state_mode`, `trimmed` flag |
| Parameters (batch, max_len, thresholds, calibration) | `config.yaml`; `operating_point.json`; `calibration.json` |
| Role of AI vs humans; who decided what | `decision_source` (model/human); PRISMA figure (human-added/removed); review queue workflow |
| Human validation: sample design, size, raters, κ | `metrics.json`; `gold_key.csv`; report §8 |
| Performance: precision, recall, F1, AUC, calibration, CIs, estimated missed records | `metrics.json`; report §8; `val01–val04` figures |
| Baselines / alternatives compared | `scorers_cross_validated` (rules, TF-IDF, stack); `compare_summary.*` |
| Failure analysis | `validation_errors.csv`; review queue A/B/C/E |
| Reproducibility (code, config, data hash, seeds) | `output/provenance/` (config without secrets, DB SHA-256), `validation.random_seed` |
| Limitations | report §9 (extend with your own) |
| Data / code availability | share code + `config.yaml`; do not redistribute publisher abstracts |
