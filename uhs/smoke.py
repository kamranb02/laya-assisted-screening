"""Laya smoke test + benchmark (version- and checkpoint-aware).

Checks (all written to output/validation/smoke_report.{json,md}):
  1. environment + installed versions (laya, torch, transformers), model load time, API/schema probe
  2. sanity accuracy on a small hand-written set incl. HARD NEGATIVES, for BOTH question styles (noul vs choice_ab)
     -> detects the `noul` label-dominance problem (upstream issue #156) and recommends the style
  3. answer stability: batch size 1 vs N
  4. context budget: share of YOUR abstracts that exceed the text budget of the loaded checkpoint
  5. throughput per batch size and an extrapolated wall-clock time for your corpus

The sanity set is tiny and written by the authors: it catches breakage, it does NOT measure real accuracy
(use `sample` -> label -> `compare` / `evaluate` for that).
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np

from .laya_compat import LayaAgent, compose_state, est_tokens, installed_versions, load_agent, probe
from .questions import Q_DETAIL, Q_UHS, QUESTIONS_ALL

# (title, abstract, is_uhs, storage_label_or_None, note)
TESTSET = [
    ("Numerical simulation of hydrogen storage in a depleted gas reservoir",
     "We simulate cyclic hydrogen injection and withdrawal in a depleted natural gas reservoir with multiphase flow, "
     "cushion gas effects and working gas recovery using a compositional reservoir simulator.", 1, "depleted_gas_reservoir", ""),
    ("Relative permeability hysteresis of hydrogen-brine systems in saline aquifers",
     "Core flooding experiments on sandstone measure hydrogen-brine drainage and imbibition relative permeability to assess "
     "residual trapping during underground hydrogen storage in saline aquifers.", 1, "aquifer", ""),
    ("Geomechanical stability of salt caverns for hydrogen storage",
     "Salt creep, cavern convergence and cyclic pressure loading are modelled to assess the mechanical integrity of salt "
     "caverns used for seasonal underground hydrogen storage.", 1, "salt_cavern", ""),
    ("Microbial hydrogen consumption during underground hydrogen storage",
     "Methanogenesis and sulfate reduction by subsurface microorganisms can consume injected hydrogen in porous reservoirs; "
     "we quantify reaction kinetics and hydrogen loss for geological storage.", 1, None, ""),
    ("Wettability of caprock minerals under hydrogen and brine",
     "Contact angle and interfacial tension measurements of hydrogen-brine on shale and clay caprock at reservoir pressure and "
     "temperature evaluate seal capacity for subsurface hydrogen storage.", 1, None, ""),
    ("Pore-network modelling of hydrogen injection in porous media",
     "A pore-scale network model quantifies capillary-controlled displacement, snap-off and trapping of hydrogen in porous "
     "rock relevant to geological hydrogen storage.", 1, None, ""),
    ("A review of geological hydrogen storage: opportunities and challenges",
     "This review summarises underground hydrogen storage in salt caverns, aquifers and depleted fields, including flow, "
     "geochemistry, microbial activity and site screening.", 1, None, ""),
    ("Site screening of depleted gas fields for large-scale hydrogen storage",
     "We assess storage capacity and containment of depleted gas fields for regional-scale hydrogen storage using field data.", 1, "depleted_gas_reservoir", ""),
    ("Sensitivity analysis of hydrogen recovery factor in a saline aquifer",
     "A global sensitivity analysis on permeability, anisotropy and injection rate identifies the main controls of hydrogen "
     "recovery in aquifer storage.", 1, "aquifer", ""),
    ("Hydrogen diffusion and mixing with cushion gas in geological storage",
     "Molecular diffusion and dispersion between hydrogen and methane cushion gas in a subsurface reservoir are simulated.", 1, None, ""),
    # ---- negatives
    ("Nickel-based catalysts for alkaline water electrolysis",
     "We develop a low-cost nickel iron catalyst improving oxygen evolution overpotential for hydrogen production.", 0, None, "easy"),
    ("Degradation of PEM fuel cell membranes",
     "Chemical degradation mechanisms of proton exchange membranes under humidity cycling are investigated.", 0, None, "easy"),
    ("Metal hydride materials for solid-state hydrogen storage",
     "Magnesium-based hydrides show improved absorption kinetics and gravimetric capacity for on-board hydrogen storage.", 0, None, "hard"),
    ("CO2 storage capacity of saline aquifers",
     "We estimate carbon dioxide storage capacity, plume migration and residual trapping in deep saline aquifers.", 0, None, "hard"),
    ("Deliverability of underground natural gas storage fields",
     "Operational performance and working gas capacity of depleted reservoir natural gas storage facilities are analysed.", 0, None, "hard"),
    ("Hydrogen embrittlement of high-strength pipeline steel",
     "Slow strain rate tests quantify hydrogen embrittlement susceptibility of X80 steel for gas transmission.", 0, None, "hard"),
    ("Techno-economic analysis of hydrogen refuelling stations",
     "Capital cost and levelised cost of hydrogen for road-vehicle refuelling stations are compared.", 0, None, "easy"),
    ("Waterflooding relative permeability in carbonate reservoirs",
     "Oil-water relative permeability and wettability alteration during low-salinity waterflooding in carbonate cores.", 0, None, "hard"),
    ("Ammonia as a hydrogen carrier for shipping",
     "Cracking of ammonia and maritime logistics for long-distance hydrogen energy transport are evaluated.", 0, None, "hard"),
    ("Type IV composite tanks for compressed hydrogen storage in vehicles",
     "Burst pressure and fatigue of carbon-fibre wound pressure vessels for 70 MPa hydrogen storage.", 0, None, "hard"),
    # ---- added in v3: 2025-2026 style topics
    ("Machine-learning surrogate for hydrogen plume migration in a saline aquifer",
     "A physics-informed neural network surrogate predicts hydrogen saturation and pressure during injection into a saline aquifer for geological hydrogen storage.", 1, "aquifer", ""),
    ("Hydrogen loss pathways during underground hydrogen storage",
     "We review microbial consumption, geochemical reactions, diffusion and leakage as pathways of hydrogen loss in subsurface storage reservoirs.", 1, None, ""),
    ("Life-cycle assessment of green hydrogen production from offshore wind",
     "Greenhouse-gas emissions of electrolytic hydrogen produced from offshore wind are assessed over the life cycle.", 0, None, "easy"),
    ("Hydrogen permeation through polymer liners of pressure vessels",
     "Permeation coefficients of polyethylene liners for compressed hydrogen vessels are measured at several temperatures.", 0, None, "hard"),
    ("Underground storage of CO2 in basalt formations",
     "Mineral carbonation and injectivity of carbon dioxide in basalt reservoirs are examined for permanent carbon storage.", 0, None, "hard"),
]


def _states():
    return [{"title": t, "abstract": a} for t, a, *_ in TESTSET]


def _p(agent, S, bs):
    return np.array([r["answers"]["uhs"]["noul"] for r in agent.predict(S, Q_UHS, bs)])


def run_smoke(cfg, paths, log, agent=None, con=None, corpus_size=None, quick=False):
    sc = cfg["screening"]
    thr = 0.5      # neutral threshold: the deployed one is only known after calibration on gold labels
    rep = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    env = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
           **installed_versions()}
    try:
        import torch
        env.update(cuda_available=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:
        env["torch_error"] = str(e)
    rep["environment"] = env

    t0 = time.perf_counter()
    base = agent or load_agent(sc, log)
    rep["model_load_seconds"] = round(time.perf_counter() - t0, 1)
    rep["checkpoint"] = {"checkpoint": base.checkpoint, "model_id": base.model_id, "max_len": base.max_len,
                         "head_max_len": base.head_max_len, "text_budget_tokens": base.text_budget}

    S = _states()
    y = np.array([t[2] for t in TESTSET])
    hard = np.array([t[4] == "hard" for t in TESTSET])
    styles = {}
    for style in ("choice_ab", "noul"):
        a = base.with_style(style)
        pr = probe(a)
        entry = {"probe": pr}
        if pr.get("ok"):
            p = _p(a, S, int(sc["batch_size"]))
            pred = (p >= thr).astype(int)
            entry.update(accuracy=round(float((pred == y).mean()), 3),
                         positives_recalled=round(float(pred[y == 1].mean()), 3),
                         hard_negatives_rejected=round(float((pred[hard] == 0).mean()), 3) if hard.any() else None,
                         separation=round(float(p[y == 1].mean() - p[y == 0].mean()), 3),
                         p_std=round(float(p.std()), 3),
                         stuck=bool(p.std() < 0.05),
                         failures=[dict(title=TESTSET[i][0], expected=int(y[i]), p_uhs=round(float(p[i]), 3))
                                   for i in range(len(TESTSET)) if pred[i] != y[i]])
            entry["probabilities"] = {TESTSET[i][0][:60]: round(float(p[i]), 3) for i in range(len(TESTSET))}
        styles[style] = entry
    rep["checks"]["styles"] = styles
    ok_styles = {k: v for k, v in styles.items() if v["probe"].get("ok") and not v.get("stuck")}
    best = max(ok_styles, key=lambda k: (ok_styles[k]["accuracy"], ok_styles[k]["separation"])) if ok_styles else None
    rep["recommended_question_style"] = best
    cur = styles.get(sc["question_style"], {})
    rep["checks"]["uhs_sanity"] = {
        "configured_style": sc["question_style"], "accuracy": cur.get("accuracy"),
        "pass": bool(cur.get("probe", {}).get("ok") and not cur.get("stuck") and (cur.get("accuracy") or 0) >= 0.8
                     and (cur.get("positives_recalled") or 0) >= 0.7),
        "note": "configured style is not the best one - consider screening.question_style: %s" % best
        if best and best != sc["question_style"] else ""}

    # ---- stability for the configured style
    a = base.with_style(sc["question_style"])
    p1 = np.array([_p(a, [s], 1)[0] for s in S])
    pb = _p(a, S, 8)
    rep["checks"]["stability"] = {"max_abs_diff_batch1_vs_batch8": round(float(np.abs(p1 - pb).max()), 4),
                                  "decision_flips_batch1_vs_batch8": int(((p1 >= thr) != (pb >= thr)).sum()),
                                  "pass": bool(np.abs(p1 - pb).max() < 0.05)}

    # ---- context budget on real abstracts
    if con is not None:
        rows = con.execute("""SELECT title, abstract FROM papers WHERE dup_of IS NULL AND eligibility='eligible'
                              AND COALESCE(abstract,'')!='' ORDER BY RANDOM() LIMIT 1000""").fetchall()
        if rows:
            toks = np.array([est_tokens((t or "") + " " + (ab or "")) for t, ab in rows])
            trimmed = np.mean([compose_state(t, ab, sc["state_mode"], base.text_budget, int(sc["max_abstract_chars"]))[1]
                               for t, ab in rows])
            rep["checks"]["context_budget"] = {
                "sampled_abstracts": len(rows), "median_est_tokens": int(np.median(toks)),
                "p90_est_tokens": int(np.percentile(toks, 90)), "text_budget_tokens": base.text_budget,
                "share_trimmed_%": round(100 * float(trimmed), 1),
                "advice": ("many abstracts exceed the text budget: compare `multilingual` (1024 tokens) with `compare`"
                           if trimmed > 0.3 else "ok")}

    # ---- throughput
    if not quick:
        pool = (S * 4)[:32]
        bench = []
        for bs in (1, 4, 8, 16):
            for label, Q in (("uhs_only", Q_UHS), ("all_questions", QUESTIONS_ALL)):
                t = time.perf_counter()
                a.predict(pool, Q, bs)
                dt = time.perf_counter() - t
                bench.append({"batch_size": bs, "questions": label, "papers_per_sec": round(len(pool) / dt, 2)})
        rep["benchmark"] = bench
        best1 = max(b["papers_per_sec"] for b in bench if b["questions"] == "uhs_only")
        bestn = max(b["papers_per_sec"] for b in bench if b["questions"] == "all_questions")
        N = corpus_size or (con.execute("SELECT COUNT(*) FROM papers WHERE dup_of IS NULL AND eligibility='eligible'").fetchone()[0]
                            if con else 4000) or 4000
        frac = 0.6
        rep["extrapolation"] = {"corpus_size": int(N), "assumed_stage2_fraction": frac,
                                "two_stage_hours": round((N / best1 + frac * N / bestn) / 3600, 1),
                                "note": "short sanity texts; real abstracts are several times longer. Real timings: output/logs/laya_performance.csv"}
    ok = all(v.get("pass", True) for v in rep["checks"].values() if isinstance(v, dict))
    rep["overall_pass"] = bool(ok)
    (paths.validation / "smoke_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    _md(rep, paths)
    log.info("SMOKE TEST %s  | recommended question_style: %s", "PASSED" if ok else "FAILED", best)
    return rep


def _md(rep, paths):
    L = ["# Laya smoke-test report", f"*{rep['timestamp']}*", "", f"**Overall: {'PASS' if rep['overall_pass'] else 'FAIL'}**",
         "", "## Environment", "```", json.dumps(rep["environment"], indent=2), "```",
         f"Checkpoint: `{json.dumps(rep['checkpoint'])}`  |  model load: {rep['model_load_seconds']} s", "",
         f"## Question styles (sanity set, threshold 0.5)  ->  recommended: **{rep['recommended_question_style']}**", "",
         "| style | accuracy | positives recalled | hard negatives rejected | separation | stuck |", "|---|---|---|---|---|---|"]
    for k, v in rep["checks"]["styles"].items():
        L.append(f"| {k} | {v.get('accuracy')} | {v.get('positives_recalled')} | {v.get('hard_negatives_rejected')} | "
                 f"{v.get('separation')} | {v.get('stuck')} |")
        for f in v.get("failures", []):
            L.append(f"|  | FAIL: {f['title']} (expected {f['expected']}, P={f['p_uhs']}) | | | | |")
    L += ["", "## Stability", "```", json.dumps(rep["checks"]["stability"], indent=2), "```"]
    if "context_budget" in rep["checks"]:
        L += ["", "## Context budget on your abstracts", "```", json.dumps(rep["checks"]["context_budget"], indent=2), "```"]
    if "benchmark" in rep:
        L += ["", "## Throughput (papers/second)", "", "| batch | questions | papers/s |", "|---|---|---|"]
        L += [f"| {b['batch_size']} | {b['questions']} | {b['papers_per_sec']} |" for b in rep["benchmark"]]
        L += ["", "```", json.dumps(rep["extrapolation"], indent=2), "```"]
    (paths.validation / "smoke_report.md").write_text("\n".join(L), encoding="utf-8")
