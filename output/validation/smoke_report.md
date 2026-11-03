# Laya smoke-test report
*2026-11-01 09:17:37*

**Overall: FAIL**

## Environment
```
{
  "python": "3.13.16",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "cpu_count": 2,
  "laya": "0.4.14",
  "torch": "2.11.0+cu130",
  "transformers": "5.21.0",
  "huggingface_hub": "1.33.0",
  "cuda_available": true,
  "gpu": "Tesla T4"
}
```
Checkpoint: `{"checkpoint": "typed-decisions", "model_id": "convaiinnovations/laya", "max_len": 1024, "head_max_len": 256, "text_budget_tokens": 768}`  |  model load: 4.3 s

## Question styles (sanity set, threshold 0.5)  ->  recommended: **noul**

| style | accuracy | positives recalled | hard negatives rejected | separation | stuck |
|---|---|---|---|---|---|
| choice_ab | 0.72 | 1.0 | 0.444 | 0.321 | False |
|  | FAIL: Nickel-based catalysts for alkaline water electrolysis (expected 0, P=0.509) | | | | |
|  | FAIL: CO2 storage capacity of saline aquifers (expected 0, P=0.532) | | | | |
|  | FAIL: Deliverability of underground natural gas storage fields (expected 0, P=0.537) | | | | |
|  | FAIL: Hydrogen embrittlement of high-strength pipeline steel (expected 0, P=0.521) | | | | |
|  | FAIL: Waterflooding relative permeability in carbonate reservoirs (expected 0, P=0.515) | | | | |
|  | FAIL: Life-cycle assessment of green hydrogen production from offshore wind (expected 0, P=0.559) | | | | |
|  | FAIL: Hydrogen permeation through polymer liners of pressure vessels (expected 0, P=0.564) | | | | |
| noul | 0.92 | 1.0 | 0.778 | 0.104 | False |
|  | FAIL: Metal hydride materials for solid-state hydrogen storage (expected 0, P=0.548) | | | | |
|  | FAIL: Deliverability of underground natural gas storage fields (expected 0, P=0.511) | | | | |

## Stability
```
{
  "max_abs_diff_batch1_vs_batch8": 0.0006,
  "decision_flips_batch1_vs_batch8": 0,
  "pass": true
}
```

## Context budget on your abstracts
```
{
  "sampled_abstracts": 1000,
  "median_est_tokens": 450,
  "p90_est_tokens": 789,
  "text_budget_tokens": 768,
  "share_trimmed_%": 11.5,
  "advice": "ok"
}
```