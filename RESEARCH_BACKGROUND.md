# Research background and design rationale (v3)

How each design decision relates to documented evidence. Sources marked ✔ were consulted in the course of building v3
(model card / abstracts / documentation pages); I did **not** read every paper in full — verify the details you cite.

## 1. The model: Laya
✔ Hugging Face model card `convaiinnovations/laya` (v0.3.20 runtime; three checkpoints: English root 512 ctx, multilingual
1024/8192, typed-decisions 1024). Statements used here:
* "Shipped checkpoints work zero-shot, but fine-tuning … is where accuracy jumps"; base English checkpoint 0.362 accuracy vs
  0.766 for the fine-tuned typed-decisions checkpoint on the *typed-decisions* benchmark, near/below the majority-class baseline
  → **do not trust zero-shot numbers; validate on your own labels and consider fine-tuning.**
* "Ships over-confident": refitting one temperature per question type and option count moves mean ECE 0.466 → 0.081 (English)
  → **calibrate on gold labels** (we use Platt scaling on logit(P), cross-validated).
* `noul` renders its options as `false:` / `true:` and the label pair can dominate the answer (issue #156); the suggested
  workaround is a two-option `choice` with neutral keys → **`question_style: choice_ab` default; both styles compared.**
* High-cardinality `choice` questions share a small option budget (`head_max_len` 192/256) → **≤ 7 options per question.**
* English checkpoint text budget ≈ 320 tokens → **smart trimming + flagging + comparison with the multilingual checkpoint.**
* `act_probability` carries no usable signal (#185) → **not used.**

✔ Independent evaluations (arXiv 2609.33843 "Laya as a Typed Probabilistic Assessor…", 2610.00346 "Benchmarking System One
decision models…", found via search) report under-confidence/over-confidence behaviour and weaker-than-zero-shot-classifier
results in some settings → consistent with the calibrate-and-validate design. (Read them before citing exact numbers.)

## 2. Reporting AI-assisted reviews
✔ PRISMA-trAIce — Transparent Reporting of AI in Systematic Literature Reviews (JMIR AI 2025;4:e80247, doi 10.2196/80247):
discipline-agnostic extension to PRISMA 2020 with items for identifying AI tools, describing human–AI interaction, reporting
performance evaluation and limitations. ✔ PRISMA-LLM (arXiv 2609.11559) and the RAISE recommendations (responsible AI in evidence
synthesis; the review team stays accountable for every decision). PRISMA 2020 (Page et al., BMJ 2021;372:n71) and PRISMA-S
(search reporting) remain the base. → `REPORTING_CHECKLIST.md`, `output/provenance/`, report "Model and settings used".

## 3. LLM-assisted title/abstract screening
✔ SESR-Eval (arXiv 2507.19027) — a dataset for evaluating LLMs on title-abstract screening; ✔ "Auxiliary uncertainty signals for
LLM-assisted systematic review screening" (arXiv 2608.14551). Take-aways used as *design principles* (not as numbers):
screening quality varies strongly by review/domain; uncertainty signals are useful for routing to humans; **recall matters more than
precision**; therefore: gold labels per project, recall-targeted threshold, review queue ordered by uncertainty/disagreement, baseline
comparison (TF-IDF + logistic regression) and stacked variants.

## 4. Data source
✔ OpenAlex: API key mandatory since 13 Feb 2026, daily free usage budget per key (documentation pages) → key from environment,
budget-aware collection, short window (2025–2026) to keep cost small. OpenAlex abstracts are often missing for some publishers →
Crossref + Semantic Scholar recovery. OpenAlex indexing lags for recent months → monthly figure with caveat, per-month rates.

## 5. UHS literature context (for seed papers and framing)
✔ 2025–2026 reviews located while building v3 (see `validation_inputs/seed_papers.txt`): Applied Energy 381 (2025) 125172 review;
Song et al., Int. J. Energy Research 2025 (depleted oil & gas reservoirs, doi 10.1155/er/9246074); salt-cavern reviews
(J. Energy Storage 2025; RSER 213:115451); "Hydrogen loss pathways…" (RSER 2026); public acceptance (Energies 2025); pore-scale
two-phase flow mini-review (Appl. Sci. 2025); machine-learning-assisted UHS appears in the literature (arXiv 2404.03222).
→ 2025–2026 themes encoded as tags/queries: hydrogen loss, ML/physics-informed surrogates, hard-rock/basalt/coal-seam settings,
blending/repurposing of gas storage, public acceptance/policy.

## 6. Why a rule-stratified gold sample
A sample stratified by the *Laya score* (v2) can only evaluate that Laya configuration and needs screening first. A sample stratified
by an independent keyword rule class (with a stratum for auto-excluded records) can be drawn before screening, is unbiased for any
classifier after design-weighting, and lets you compare checkpoints/styles/fine-tuned models on the same labels.

## 7. Suggested paper framing
"AI-assisted scoping review / bibliometric mapping of underground hydrogen storage research 2025–2026" — contributions:
(i) a validated, recall-targeted LLM-assisted screening workflow with calibrated probabilities and human-in-the-loop closure;
(ii) a map of settings × methods × scales × processes with absolute and relative gaps; (iii) topic shifts 2025 → 2026.
Avoid the label "systematic review" unless you add a registered protocol, a second database and dual human screening.
