# Fine-tuning Laya on your gold labels (optional, recommended if `compare` shows weak results)

The Laya model card states that base checkpoints are a fast *base to specialise* and that in-domain fine-tuning is where accuracy
improves; it provides a Kaggle notebook (free 2×T4) that builds a dataset, trains, fits calibration temperatures, evaluates and
can push to the Hub: https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb

1. Label ≥ 400 gold papers (more is better for fine-tuning; keep a part untouched).
2. `python run.py export-finetune` → `output/validation/finetune_gold.jsonl`. One record per labeled paper:
   `{"gold_id", "split": train|dev|test, "state": {title, abstract}, "questions": {…Laya format…}, "answers": {…}}`.
   Splits are deterministic (hash of the OpenAlex id). **Never train on `test`.**
3. In the official notebook replace its dataset-building step by reading this file. *The exact training-record schema is defined by the
   notebook/SDK, which I could not execute here — adapt the few lines that map `state/questions/answers`.*
4. Fit temperatures on `dev`, report on `test`.
5. Put the resulting repo id or local path in `config.yaml → compare.variants` (`checkpoint: custom`, `model_id: …`), run
   `python run.py compare`; if it wins set `screening.checkpoint: custom`, `screening.model_id: …`, then
   `python run.py screen --rescreen` and `python run.py evaluate`.
6. In the paper report the fine-tuned model as a *different model* than base Laya, with its own validation numbers, and keep the
   test split out of calibration/threshold selection (use a fresh gold sample for the final `evaluate` if you can).
