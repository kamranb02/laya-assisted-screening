"""Offline end-to-end test (no internet, no Laya weights needed).

    python -m tests.test_offline

A mocked OpenAlex/Crossref/Semantic-Scholar client and fake Laya agents (BOTH API shapes: with and without
`predict_batch`; BOTH question styles) exercise the whole chain:
collect -> process (dedup, abstract recovery, rules) -> gold sample (rule-stratified) -> labels -> screen ->
evaluate (calibration, baselines, CIs, operating point) -> analyze -> review ingestion -> compare -> fine-tune export
-> seed recall -> configuration-mix protection.
It tests the PIPELINE LOGIC only; it says nothing about Laya's real accuracy (use `run.py smoke`, `compare`, `evaluate`).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

from uhs import analyze, collect, export, figures, laya_compat, metrics as M, process, report, screen, validate  # noqa: E402
from uhs.common import connect, get_logger, get_meta, load_config, make_paths  # noqa: E402
from uhs.laya_compat import AgentPool, LayaAgent, compose_state  # noqa: E402

random.seed(1)
POS = [
    ("Hydrogen storage in depleted gas reservoirs: multiphase flow simulation",
     "Cyclic hydrogen injection and withdrawal in a depleted gas reservoir is simulated with CMG GEM; cushion gas, relative permeability and capillary pressure sensitivity analysis. " * 2),
    ("Underground hydrogen storage in saline aquifers: core flooding of sandstone",
     "Hydrogen-brine relative permeability and contact angle in sandstone cores for subsurface hydrogen storage in aquifers; residual trapping and hysteresis. Experiments were run in the laboratory."),
    ("Salt cavern hydrogen storage geomechanics",
     "Salt creep, cavern convergence and cyclic loading of salt caverns for underground hydrogen storage; thermal effects and well integrity."),
    ("Methanogenesis during geological hydrogen storage",
     "Microbial methanogenesis and sulfate reduction consume hydrogen in subsurface porous reservoirs; geochemistry and reactive transport; hydrogen loss."),
    ("Machine-learning surrogate for underground hydrogen storage",
     "A physics-informed neural network surrogate for hydrogen plume migration in an aquifer used for geological hydrogen storage."),
]
NEG = [
    ("PEM fuel cell degradation", "Membrane degradation in proton exchange fuel cells under humidity cycling."),
    ("CO2 storage in saline aquifers", "Carbon dioxide plume migration and residual trapping in deep saline aquifers."),
    ("Hydrogen embrittlement of steel", "Hydrogen embrittlement susceptibility of pipeline steel in gas transmission."),
    ("Deep learning for image segmentation", "A neural network for medical imaging."),
]
TRUTH = {}


def inv(text):
    words = text.split()
    d = {}
    for i, w in enumerate(words):
        d.setdefault(w, []).append(i)
    return d


def make_pool(n=420):
    pool = []
    for i in range(n):
        if random.random() < .5:
            t, a = random.choice(POS)
            truth = 1
        else:
            t, a = random.choice(NEG)
            truth = 0
        t = f"{t} #{i}"
        year = random.choice([2025, 2025, 2026, 2026, 2024])
        month = random.randint(1, 9) if year == 2026 else random.randint(1, 12)
        no_abs = random.random() < .15
        TRUTH[t] = truth
        pool.append({"id": f"https://openalex.org/W{1000 + i}", "doi": f"https://doi.org/10.1000/test.{i}", "title": t,
                     "publication_year": year, "publication_date": f"{year}-{month:02d}-15", "type": "article",
                     "language": "en", "cited_by_count": random.randint(0, 20), "is_retracted": False,
                     "primary_location": {"source": {"display_name": random.choice(["Int J Hydrogen Energy", "Fuel", "Energy"]),
                                                     "host_organization_name": "Elsevier"}, "landing_page_url": "http://x"},
                     "best_oa_location": None, "open_access": {"is_oa": False, "oa_status": "closed"},
                     "authorships": [{"author": {"display_name": f"A{i}"}, "countries": random.sample(["US", "CN", "GB", "DE"], random.randint(1, 2)),
                                      "institutions": [{"display_name": random.choice(["MIT", "Tsinghua", "Imperial"])}]}],
                     "abstract_inverted_index": None if no_abs else inv(a), "keywords": [{"display_name": "hydrogen"}],
                     "primary_topic": {"display_name": "x"}, "_abstract": a})
    # a preprint duplicate of record 0 (same title, no DOI) and an erratum
    d = dict(pool[0]); d["id"] = "https://openalex.org/W9999"; d["doi"] = None; d["type"] = "preprint"; pool.append(d)
    e = dict(pool[1]); e["id"] = "https://openalex.org/W9998"; e["title"] = "Erratum to something"; e["type"] = "erratum"; pool.append(e)
    return pool


class FakeClient:
    api_key = "TESTKEY"

    def __init__(self, pool, budget_after=None):
        self.pool, self.calls, self.budget_after = pool, 0, budget_after
        self.by_doi = {w["doi"].replace("https://doi.org/", ""): w for w in pool if w["doi"]}

    def get_json(self, url, params=None):
        if "crossref" in url:
            doi = url.split("/works/")[1]
            w = self.by_doi.get(doi)
            if w and int(doi.split(".")[-1]) % 2 == 0:
                return {"message": {"abstract": "<jats:p>" + w["_abstract"] + "</jats:p>"}}
            return None
        self.calls += 1
        if self.budget_after and self.calls > self.budget_after:
            raise collect.BudgetExhausted("daily budget used up")
        cur = 0 if params["cursor"] == "*" else int(params["cursor"])
        per = params["per-page"]
        page = self.pool[cur:cur + per]
        nxt = str(cur + per) if cur + per < len(self.pool) else None
        return {"meta": {"count": len(self.pool), "next_cursor": nxt}, "results": [{k: v for k, v in w.items() if k != "_abstract"} for w in page]}

    def post_json(self, url, json=None, params=None, headers=None):
        out = []
        for ident in json["ids"]:
            w = self.by_doi.get(ident.replace("DOI:", ""))
            out.append({"abstract": w["_abstract"]} if w and int(ident.split(".")[-1]) % 2 == 1 else None)
        return out


# ------------------------------------------------------------------ fake Laya
def _p_uhs(state):
    txt = (state["title"] + " " + state["abstract"]).lower()
    pos = ("hydrogen" in txt) and any(k in txt for k in ("reservoir", "aquifer", "cavern", "subsurface", "geological", "methanogenesis"))
    neg = any(k in txt for k in ("fuel cell", "carbon dioxide", "embrittlement", "medical"))
    h = int(hashlib.md5(txt.encode()).hexdigest(), 16) % 1000 / 1000.0
    if pos and not neg:
        return 0.55 + 0.4 * h
    return 0.02 + 0.3 * h if not pos else 0.4 + 0.1 * h


def _answer(state, qdef):
    p = _p_uhs(state)
    txt = (state["title"] + " " + state["abstract"]).lower()
    if qdef["type"] == "noul":
        return {"noul": p}
    keys = list(qdef["criteria"])
    if keys == ["A", "B"]:
        return {"choice": "A" if p >= .5 else "B", "probabilities": {"A": p, "B": 1 - p}}
    pick = keys[0]
    for k in keys:
        if k.split("_")[0] in txt:
            pick = k
    probs = {k: (0.7 if k == pick else 0.3 / (len(keys) - 1)) for k in keys}
    return {"choice": pick, "probabilities": probs}


class FakeRawBatch:
    """API shape 1: predict_batch (as in laya 0.3.x)."""
    cfg = {"max_len": 512, "head_max_len": 192}
    device = "cpu"

    def predict_batch(self, states, questions, batch_size=8, max_len=512):
        return [{"answers": {q: _answer(s, d) for q, d in questions.items()}, "usage": {"input_tokens": 40 * len(questions)}}
                for s in states]


class FakeRawSingle:
    """API shape 2: only predict (older/newer runtime without predict_batch)."""
    cfg = {"max_len": 1024, "head_max_len": 256}
    device = "cpu"

    def predict(self, state, questions):
        return {"answers": {q: _answer(state, d) for q, d in questions.items()}, "usage": {"input_tokens": 10}}


def fake_pool(cfg, raw_cls=FakeRawBatch, style=None):
    sc = cfg["screening"]
    a = LayaAgent(raw_cls(), "english", "fake/laya", style or sc["question_style"], batch_size=8)
    return AgentPool(sc, None, agents={"english": a})


# ------------------------------------------------------------------- the test
def main():
    tmp = Path(tempfile.mkdtemp(prefix="uhs_test_"))
    cfg = load_config()
    cfg["paths"]["data_dir"], cfg["paths"]["output_dir"] = str(tmp / "data"), str(tmp / "output")
    cfg["project"]["end_date"] = "2026-09-30"
    cfg["_window"] = ("2025-01-01", "2026-09-30")
    cfg["_start_year"], cfg["_end_year"] = 2025, 2026
    cfg["queries"] = cfg["queries"][:3]
    cfg["validation"].update(sample_size=160, bootstrap_reps=200, seed_file=str(tmp / "seeds.txt"))
    cfg["abstract_enrichment"]["semantic_scholar"]["sleep_seconds"] = 0
    cfg["abstract_enrichment"]["crossref"]["sleep_seconds"] = 0
    cfg["openalex"]["sleep_seconds"] = 0
    cfg["analysis"]["formats"] = ["png"]
    cfg["analysis"]["dpi"] = 80
    paths = make_paths(cfg)
    log = get_logger(paths)
    con = connect(paths)
    pool = make_pool()

    # ---- 1. collection incl. budget exhaustion + resume
    cl = FakeClient(pool, budget_after=3)
    ok = collect.collect(con, cfg, log, client=cl)
    assert ok is False, "budget exhaustion must stop the run"
    assert con.execute("SELECT COUNT(*) FROM query_state WHERE done=0").fetchone()[0] >= 1
    cl = FakeClient(pool)
    assert collect.collect(con, cfg, log, client=cl), "resume must complete"
    n = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert n == len(pool), (n, len(pool))
    print(f"[ok] collection incl. budget stop + resume: {n} records")

    # ---- 2. process: dedup, abstract recovery, rules
    missing0 = con.execute("SELECT COUNT(*) FROM papers WHERE COALESCE(abstract,'')=''").fetchone()[0]
    process.dedup(con, log)
    f1 = collect.enrich_crossref(con, cfg, log, client=cl)
    f2 = collect.enrich_semantic_scholar(con, cfg, log, client=cl)
    process.dedup(con, log)
    stats = process.apply_rules(con, cfg, log)
    assert f1 + f2 > 0 and f1 + f2 <= missing0
    assert con.execute("SELECT COUNT(*) FROM papers WHERE dup_of IS NOT NULL").fetchone()[0] >= 1
    assert stats.get("type:erratum", 0) == 1 and stats.get("year_out_of_range", 0) > 0
    print(f"[ok] process: abstracts recovered crossref={f1} s2={f2}; eligibility={stats}")

    # ---- 3. gold sample BEFORE screening (rule-stratified), label from truth
    sheet = validate.make_sample(con, cfg, paths, log, n=160)
    wb = load_workbook(sheet)
    ws = wb["Labels"]
    hdr = [c.value for c in ws[1]]
    ci = {h: i + 1 for i, h in enumerate(hdr)}
    rng = random.Random(5)
    for r in range(2, ws.max_row + 1):
        t = ws.cell(r, ci["title"]).value
        base = t.split(" #")[0]
        truth = TRUTH.get(t, 1 if t.startswith("Hydrogen storage") else 0)
        if rng.random() < .04:
            truth = 1 - truth                      # a few annotator "mistakes"
        ws.cell(r, ci["human_relevant"]).value = truth
        if truth:
            ws.cell(r, ci["human_storage_type"]).value = "depleted_gas_reservoir" if "depleted" in t else "general_or_unclear"
            ws.cell(r, ci["human_method"]).value = "numerical"
            ws.cell(r, ci["human_scale"]).value = "reservoir"
    wb.save(sheet)
    f2 = validate.make_sample(con, cfg, paths, log, second_rater=True)
    wb2 = load_workbook(f2)
    w2 = wb2["Labels"]
    c2 = {c.value: i + 1 for i, c in enumerate(w2[1])}
    rng2 = random.Random(9)
    for r in range(2, w2.max_row + 1):
        t = w2.cell(r, c2["title"]).value
        truth = TRUTH.get(t, 1 if t.startswith("Hydrogen storage") else 0)
        w2.cell(r, c2["human_relevant"]).value = 1 - truth if rng2.random() < .05 else truth
    wb2.save(f2)
    key = pd.read_csv(paths.validation / validate.KEY)
    assert {"strong", "likely", "unlikely", "none"} & set(key.stratum) and (key.stratum_size >= key.stratum_n).all()
    print(f"[ok] gold sample: {len(key)} papers, strata {key.stratum.value_counts().to_dict()}")

    # ---- 4. screening with BOTH API shapes / styles, config-mix protection
    pool_a = fake_pool(cfg, FakeRawBatch, "choice_ab")
    screen.screen(con, cfg, paths, log, pool=pool_a)
    tags = {r[0] for r in con.execute("SELECT DISTINCT model_tag FROM laya")}
    assert len(tags) == 1, tags
    assert con.execute("SELECT COUNT(*) FROM laya WHERE stage=2").fetchone()[0] > 0
    row = con.execute("SELECT answers_json FROM laya WHERE stage=2 LIMIT 1").fetchone()[0]
    ans = json.loads(row)
    assert ans["uhs"]["type"] == "noul" and ans["geochemistry"]["type"] == "noul" and ans["storage_type"]["type"] == "choice"
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["screening"]["question_style"] = "noul"
    pool_b = fake_pool(cfg2, FakeRawSingle, "noul")
    try:
        screen.screen(con, cfg2, paths, log, pool=pool_b)
        raise AssertionError("mixing configurations must be refused")
    except RuntimeError as e:
        assert "different configuration" in str(e)
    screen.screen(con, cfg2, paths, log, pool=pool_b, rescreen=True, limit=40)     # predict-only API + noul style
    assert any(r[0].startswith("laya_backup_") for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    screen.screen(con, cfg2, paths, log, pool=pool_b, rescreen=False)              # completes the rest, same tag
    # back to the main configuration for the rest of the test
    screen.screen(con, cfg, paths, log, pool=pool_a, rescreen=True)
    print("[ok] screening: two API shapes, both question styles, mix protection, backup on --rescreen")

    # ---- 5. evaluate: calibration, baselines, CIs, operating point
    res = validate.evaluate(con, cfg, paths, log)
    op = json.loads((paths.validation / "operating_point.json").read_text())
    sc = res["scorers_cross_validated"]
    assert sc["laya_raw"]["auc_w"] > 0.9, sc["laya_raw"]
    assert op["scorer"] == "laya_cal" and 0 < op["threshold"] < 1
    assert res["operating_point"]["recall"] >= 0.8
    assert (paths.validation / "val01_threshold_sweep.png").exists() and (paths.validation / "calibration.json").exists()
    assert "tfidf" in sc and "stack" in sc
    assert res["inter_rater"] is not None and res["inter_rater"]["kappa_relevance"] > 0.6
    assert res["categorical"], "categorical evaluation expected"
    assert con.execute("SELECT COUNT(*) FROM manual WHERE source='gold'").fetchone()[0] >= 100
    print(f"[ok] evaluate: AUC={sc['laya_raw']['auc_w']}, op={op['scorer']}@{op['threshold']}, recall={res['operating_point']['recall']}, "
          f"kappa(rater1,2)={res['inter_rater']['kappa_relevance']}")

    # ---- 6. analyze with validated operating point + human overrides
    df = analyze.build_frame(con, cfg, paths, log)
    P = analyze.prisma_counts(con, df, cfg)
    assert P["validated"] and P["scorer"] == "laya_cal"
    assert P["human_included"] + P["human_excluded"] > 0
    assert (df.decision_source == "human").sum() > 0
    assert P["included"] == df.decision.eq("included").sum()
    T = analyze.make_tables(con, df, cfg)
    assert {"yearly", "monthly", "topic_shift", "coverage_gaps", "review_queue", "top_countries_fractional"} <= set(T)
    Y = T["yearly"]
    assert bool(Y.loc[Y.publication_year == 2026, "partial_year"].iloc[0]) and not bool(Y.loc[Y.publication_year == 2025, "partial_year"].iloc[0])
    assert set(df[df.decision.isin(["included", "excluded"])].publication_year.dropna().astype(int)) <= {2025, 2026}
    figures.Figs(cfg, paths, log).all(df, T, P, None)
    export.write_outputs(df, T, P, cfg, paths, log)
    (paths.root / "x").parent  # noqa: B018
    # seeds
    inc_titles = df[df.decision == "included"].title.head(3).tolist()
    Path(cfg["validation"]["seed_file"]).write_text("# seeds\n" + "\n".join(inc_titles) + "\nA paper that was never collected\n", encoding="utf-8")
    seed = validate.recall_check(con, cfg, paths, log)
    assert seed["seeds"] == 4 and seed["collected"] == 3
    report.build_report(df, T, P, cfg, paths, log, con=con)
    html = (paths.report / "UHS_Literature_Report.html").read_text(encoding="utf-8")
    assert "Validated against" in html and "partial year" in html
    print(f"[ok] analyze/report: included={P['included']} (human {P['human_included']}), review queue={P['review_queue_size']}")

    # ---- 7. review ingestion changes decisions
    rq = T["review_queue"]
    assert "human_decision" in rq
    if len(rq) == 0:     # the toy data has no rule/model disagreement: use model-included rows instead
        rq = df[(df.decision == "included") & (df.decision_source == "model")].head(5)[["openalex_id"]].assign(human_decision="")
    q = rq.head(5).copy()
    q["human_decision"] = 0
    f = paths.tables / "rq_filled.csv"
    q.to_csv(f, index=False)
    validate.ingest_review(con, f, log)
    df2 = analyze.build_frame(con, cfg, paths, log)
    assert (df2.set_index("openalex_id").loc[q.openalex_id, "decision"] == "excluded").all()
    print("[ok] review ingestion overrides decisions")

    # ---- 8. compare variants and fine-tune export
    variants = {"english_ab": {"checkpoint": "english", "question_style": "choice_ab"},
                "english_noul": {"checkpoint": "english", "question_style": "noul"}}
    cfg["compare"]["variants"] = variants

    def loader(sc, log_, checkpoint=None, style=None, max_len=None, model_id=None, head_max_len=None):
        return LayaAgent(FakeRawBatch(), checkpoint, "fake", style, max_len=max_len or 0)
    R = validate.compare(con, cfg, paths, log, loader=loader)
    assert len(R) == 2 and R.auc_w.min() > 0.85
    jl = validate.export_finetune(con, cfg, paths, log)
    first = json.loads(jl.read_text(encoding="utf-8").splitlines()[0])
    assert {"state", "questions", "answers", "split"} <= set(first)
    print(f"[ok] compare ({R.auc_w.round(3).tolist()}) and fine-tune export")

    # ---- 9. unit checks
    st, trimmed, est = compose_state("T", ("Sentence one is here. " * 80) + "Final conclusion sentence.", "smart_trim", 100, 6000)
    assert trimmed == 1 and "Final conclusion" in st["abstract"] and est <= 130
    st, trimmed, _ = compose_state("T", "Short abstract.", "smart_trim", 300, 6000)
    assert trimmed == 0
    a, b = M.fit_platt(np.array([.9, .8, .7, .3, .2, .1]), np.array([1, 1, 1, 0, 0, 0]))
    assert a > 0
    assert M.kappa(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0])) == 1.0
    assert laya_compat.est_tokens("a" * 400) == 101
    print("[ok] unit checks (smart_trim, platt, kappa)")
    # ---- 10. smoke test with a fake agent, provenance bundle, live dashboard snapshot
    from uhs import smoke
    import run as runmod
    rep = smoke.run_smoke(cfg, paths, log, agent=LayaAgent(FakeRawBatch(), "english", "fake", "choice_ab"), con=con, quick=True)
    assert rep["overall_pass"] and rep["recommended_question_style"] in ("choice_ab", "noul"), rep["checks"]["uhs_sanity"]
    assert "context_budget" in rep["checks"]
    runmod.do_freeze(con, cfg, paths, log)
    assert (paths.provenance / "provenance.json").exists()
    from uhs import live
    live.run_live(cfg, paths, log, port=0, interval=1, open_browser=False, once=True)
    runmod.do_status(con, cfg, paths, log)
    print("[ok] smoke (fake agent), provenance bundle, live snapshot, status")
    print(f"\nALL OFFLINE TESTS PASSED  (workdir: {tmp})")


if __name__ == "__main__":
    main()
