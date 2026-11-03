"""Stage 6 - measuring how good the classification really is (and choosing the operating point from evidence).

Workflow
  1. `sample`   -> BLIND labeling sheet. Default design: stratified by keyword-rule class, so it does NOT depend on
                   Laya and can be drawn BEFORE screening. The same gold labels then evaluate any Laya checkpoint,
                   question style or fine-tuned model (`compare`) and calibrate/validate the deployed one (`evaluate`).
  2. You label ~400 papers in Excel (dropdowns provided). Optional: `sample --second-rater` writes a blind copy of a
     share of the sheet for a second reviewer (inter-rater kappa + adjudication list).
  3. `compare`  -> screens ONLY the gold papers with several Laya variants (checkpoint x question style) and ranks them
                   by precision at the recall target.
  4. `evaluate` -> weighted precision / recall / F1 / AUC / ECE with bootstrap CIs for: raw Laya, calibrated Laya,
                   keyword rules, a TF-IDF logistic baseline and a stacked model (all cross-validated), chooses the
                   threshold that reaches the recall target, and writes operating_point.json, which `analyze` applies.
  5. `recall`   -> checks that landmark (seed) papers were collected and included.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from openpyxl import Workbook, load_workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from . import metrics as M  # noqa: E402
from .common import get_meta, norm_doi, norm_title, now, set_meta  # noqa: E402
from .laya_compat import compose_state  # noqa: E402
from .questions import BINARY_DETAIL, Q_DETAIL, Q_UHS, QUESTIONS_ALL, question_hash  # noqa: E402
from .scorers import OP_FILE, rule_dummies, stack_features, tfidf_text  # noqa: E402

SHEET, KEY, SHEET2 = "gold_labeling_sheet.xlsx", "gold_key.csv", "gold_rater2_sheet.xlsx"
ADJ = "adjudication.csv"
CATS = {"human_storage_type": ("laya_storage_type", "rule_storage", list(Q_DETAIL["storage_type"]["criteria"])),
        "human_method": ("laya_method", "rule_method", list(Q_DETAIL["method"]["criteria"])),
        "human_scale": ("laya_scale", "rule_scale", list(Q_DETAIL["scale"]["criteria"]))}
FLAG_COLS = {f"human_{q}": q for q in BINARY_DETAIL}
RULE_SCORE = {"strong": 1.0, "likely": 0.6, "unlikely": 0.2, "none": 0.0}
GRID = np.round(np.arange(0.01, 0.991, 0.01), 2)


# ------------------------------------------------------------------ sampling
def _alloc(sizes: dict, n: int, share: dict) -> dict:
    """Integer allocation per stratum, capped at the stratum size, leftovers redistributed."""
    tot = sum(share.get(k, 0) for k in sizes) or 1
    want = {k: int(round(n * share.get(k, 0) / tot)) for k in sizes}
    got = {k: min(want[k], sizes[k]) for k in sizes}
    left = n - sum(got.values())
    while left > 0:
        room = {k: sizes[k] - got[k] for k in sizes if sizes[k] > got[k]}
        if not room:
            break
        for k in sorted(room, key=lambda x: -room[x]):
            if left <= 0:
                break
            got[k] += 1
            left -= 1
    return got


def _sample_rules(con, cfg, n, seed):
    base = pd.read_sql_query(
        """SELECT openalex_id,title,abstract,doi,publication_year,eligibility,exclude_reason,rule_uhs
           FROM papers WHERE dup_of IS NULL""", con)
    elig = base[base.eligibility == "eligible"].copy()
    elig["stratum"] = elig.rule_uhs.fillna("none")
    auto = base[(base.eligibility == "auto_excluded") & (base.exclude_reason == "no_hydrogen_term")].copy()
    auto["stratum"] = "auto_excluded_no_hydrogen"
    pool = pd.concat([elig, auto])
    sizes = pool.stratum.value_counts().to_dict()
    got = _alloc(sizes, n, cfg["validation"].get("allocation", {}))
    parts = []
    for k, g in pool.groupby("stratum"):
        if got.get(k, 0) > 0:
            s = g.sample(got[k], random_state=seed)
            parts.append(s.assign(stratum_size=len(g), stratum_n=len(s)))
    return pd.concat(parts)


def _sample_laya(con, cfg, n, seed, paths=None):
    from .analyze import build_frame
    df = build_frame(con, cfg, None)
    scr = df[df.uhs_prob.notna() & df.dup_of.isna() & df.eligibility.eq("eligible")].copy()
    edges = [0, .1, .3, .5, .7, .9, 1.0001]
    scr["stratum"] = pd.cut(scr.uhs_prob, edges, right=False).astype(str)
    strata = [g for _, g in scr.groupby("stratum")]
    per = max(1, n // max(len(strata), 1))
    parts = [g.sample(min(per, len(g)), random_state=seed).assign(stratum_size=len(g), stratum_n=min(per, len(g)))
             for g in strata]
    return pd.concat(parts)


def _style_sheet(ws, widths):
    hf = PatternFill("solid", fgColor="1F4E79")
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), hf
    for i, w in enumerate(widths):
        ws.column_dimensions[ws.cell(row=1, column=i + 1).column_letter].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "E2"


def _add_validation(ws, header, last):
    """Dropdowns for the human_* columns."""
    cols = {c.value: c.column_letter for c in ws[1]}
    lists = {"human_relevant": "1,0"}
    for h, (_, _, labs) in CATS.items():
        lists[h] = ",".join(labs)
    for h in FLAG_COLS:
        lists[h] = "1,0"
    for h, opts in lists.items():
        if h in cols:
            dv = DataValidation(type="list", formula1=f'"{opts}"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{cols[h]}2:{cols[h]}{last}")


INSTRUCTIONS = [
    "BLIND LABELING SHEET - model scores are deliberately hidden.",
    "human_relevant: 1 = the paper is SUBSTANTIALLY about underground / geological / subsurface hydrogen storage "
    "(including its rock-fluid, geochemical, microbial, geomechanical, integrity or economic aspects), 0 = not.",
    "  hydrogen production, electrolysers, fuel cells, tanks, metal hydrides, CO2-only or natural-gas-only storage = 0.",
    "human_storage_type / human_method / human_scale: fill ONLY when human_relevant = 1 (dropdowns).",
    "  storage_type = principal setting; general_or_unclear for reviews of several settings. method = MAIN method.",
    "human_<topic> columns (optional): 1/0 'substantially studies this topic'; fill only if you have time - they validate the topic flags.",
    "Judge from title + abstract only (exactly what the model sees). Use notes for doubtful cases.",
    "You may save partially and continue later. Do not reorder/delete columns A-B.",
    "When done run:  python run.py compare   (optional)   then   python run.py evaluate",
]


def make_sample(con, cfg, paths, log, n=None, force=False, design=None, second_rater=False):
    n = int(n or cfg["validation"]["sample_size"])
    seed = int(cfg["validation"]["random_seed"])
    sheet, key = paths.validation / SHEET, paths.validation / KEY
    if second_rater:
        return _second_rater(cfg, paths, log, seed)
    if sheet.exists() and not force:
        raise FileExistsError(f"{sheet} already exists (your labels!). Use --force to overwrite.")
    design = design or cfg["validation"].get("design", "rules")
    S = (_sample_rules if design == "rules" else _sample_laya)(con, cfg, n, seed)
    S = S.sample(frac=1, random_state=seed).reset_index(drop=True)
    S["gold_id"] = [f"G{i + 1:03d}" for i in range(len(S))]
    S["design"] = design
    S[["gold_id", "openalex_id", "stratum", "stratum_size", "stratum_n", "design"]].to_csv(
        key, index=False, encoding="utf-8-sig")
    wb = Workbook()
    ws = wb.active
    ws.title = "Labels"
    hdr = (["gold_id", "openalex_id", "year", "title", "abstract", "doi", "human_relevant", "human_storage_type",
            "human_method", "human_scale"] + list(FLAG_COLS) + ["notes"])
    ws.append(hdr)
    for r in S.itertuples():
        ws.append([r.gold_id, r.openalex_id, int(r.publication_year) if pd.notna(r.publication_year) else None,
                   r.title, (r.abstract or "")[:32000], r.doi] + [None] * (len(hdr) - 6))
    _style_sheet(ws, [8, 26, 7, 50, 90, 28, 14, 22, 16, 12] + [12] * len(FLAG_COLS) + [30])
    _add_validation(ws, hdr, len(S) + 1)
    ins = wb.create_sheet("Instructions")
    for line in INSTRUCTIONS:
        ins.append([line])
    ins.column_dimensions["A"].width = 130
    wb.save(sheet)
    log.info("Labeling sheet with %d papers (design=%s, strata=%s) written to %s", len(S), design,
             S.stratum.value_counts().to_dict(), sheet)
    return sheet


def _second_rater(cfg, paths, log, seed):
    key = pd.read_csv(paths.validation / KEY)
    s1 = pd.read_excel(paths.validation / SHEET, sheet_name="Labels")
    k = max(20, int(len(key) * float(cfg["validation"].get("second_rater_fraction", 0.25))))
    pick = key.sample(min(k, len(key)), random_state=seed + 1).gold_id.tolist()
    sub = s1[s1.gold_id.isin(pick)].sample(frac=1, random_state=seed + 2)
    wb = Workbook()
    ws = wb.active
    ws.title = "Labels"
    hdr = ["gold_id", "openalex_id", "year", "title", "abstract", "doi", "human_relevant", "human_storage_type",
           "human_method", "human_scale", "notes"]
    ws.append(hdr)
    for r in sub.itertuples():
        ws.append([r.gold_id, r.openalex_id, r.year, r.title, r.abstract, r.doi, None, None, None, None, None])
    _style_sheet(ws, [8, 26, 7, 50, 90, 28, 14, 22, 16, 12, 30])
    _add_validation(ws, hdr, len(sub) + 1)
    ins = wb.create_sheet("Instructions")
    for line in INSTRUCTIONS[:5] + INSTRUCTIONS[6:7]:
        ins.append([line])
    ins.column_dimensions["A"].width = 130
    f = paths.validation / SHEET2
    wb.save(f)
    log.info("Second-rater sheet (%d papers, blind) written to %s", len(sub), f)
    return f


# ------------------------------------------------------------------ gold data
def _loads(s):
    try:
        return json.loads(s) if isinstance(s, str) and s else {}
    except Exception:
        return {}


def _read_labels(paths):
    g = pd.read_excel(paths.validation / SHEET, sheet_name="Labels")
    g["human_relevant"] = pd.to_numeric(g["human_relevant"], errors="coerce")
    adj = paths.validation / ADJ
    if adj.exists():
        a = pd.read_csv(adj)
        a = a[pd.to_numeric(a.get("final_relevant"), errors="coerce").notna()]
        mp = dict(zip(a.gold_id, pd.to_numeric(a.final_relevant)))
        g["human_relevant"] = [mp.get(i, v) for i, v in zip(g.gold_id, g.human_relevant)]
    return g


def gold_frame(con, cfg, paths, log, need_laya=True):
    """Labeled gold papers + keyword rules + Laya answers + design weights."""
    g = _read_labels(paths)
    g = g.dropna(subset=["human_relevant"])
    key = pd.read_csv(paths.validation / KEY)
    sql = """SELECT p.openalex_id,p.eligibility,p.dup_of,p.exclude_reason,p.rule_uhs,p.rule_storage,p.rule_method,
                    p.rule_scale,p.title AS p_title,p.abstract AS p_abstract,l.stage,l.uhs_prob,l.answers_json,
                    l.trimmed,l.near_max_len
             FROM papers p LEFT JOIN laya l ON l.openalex_id=p.openalex_id"""
    P = pd.read_sql_query(sql, con)
    m = (g.drop(columns=["year"], errors="ignore").merge(key.drop(columns=["gold_id"]), on="openalex_id")
          .merge(P, on="openalex_id", how="left"))
    m["y"] = m.human_relevant.astype(int)
    m["reached"] = m.eligibility.eq("eligible") & m.dup_of.isna()
    unscreened = m.reached & m.uhs_prob.isna()
    if need_laya and unscreened.any():
        frac = unscreened.mean()
        log.warning("%d labeled gold papers have no Laya result yet (%.0f%%) - run `screen` first; they are excluded.",
                    int(unscreened.sum()), 100 * frac)
        if frac > 0.05:
            raise RuntimeError("More than 5% of the labeled gold papers are not screened yet: run `python run.py screen`.")
        m = m[~unscreened].copy()
    m["raw"] = np.where(m.reached, m.uhs_prob, 0.0)
    nh = m.groupby("stratum").stratum.transform("size")
    m["w"] = m.stratum_size / nh                       # design weight with the number actually labeled
    A = m.answers_json.map(_loads)
    for q, d in Q_DETAIL.items():
        if d["type"] == "choice":
            m[f"laya_{q}"] = A.map(lambda a, q=q: (a.get(q) or {}).get("choice"))
            m[f"laya_{q}_conf"] = A.map(lambda a, q=q: ((a.get(q) or {}).get("probabilities") or {}).get(
                (a.get(q) or {}).get("choice"), np.nan))
        else:
            m[f"laya_{q}"] = A.map(lambda a, q=q: (a.get(q) or {}).get("noul", np.nan))
    m["title"] = m["title"].fillna(m.p_title)
    m["abstract"] = m["abstract"].fillna(m.p_abstract)
    return m.reset_index(drop=True)


# ----------------------------------------------------------------- scorers
def _fit_models(m, fit_idx):
    """Fit every learned scorer on rows fit_idx (reached only); return a callable producing scores for any index."""
    fi = np.array([i for i in fit_idx if m.reached.iloc[i]])
    out = {}
    y, w, raw = m.y.to_numpy()[fi], m.w.to_numpy()[fi], m.raw.to_numpy()[fi]
    ok_cal = len(fi) >= 15 and y.min() != y.max()
    if ok_cal:
        a, b = M.fit_platt(raw, y, w)
        out["laya_cal"] = {"a": a, "b": b}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        if ok_cal:
            X = tfidf_text(m.title.iloc[fi], m.abstract.iloc[fi])
            pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
                                 LogisticRegression(C=4.0, max_iter=2000))
            pipe.fit(X, y, logisticregression__sample_weight=w / w.mean())
            out["tfidf"] = pipe
            F = stack_features(raw, m.rule_uhs.iloc[fi].fillna("none"))
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(F, y, sample_weight=w / w.mean())
            out["stack"] = {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    except ImportError:
        pass
    return out


def _apply(models, m, idx):
    """Scores of every scorer on rows idx (unreached rows score 0)."""
    sub = m.iloc[idx]
    reached = sub.reached.to_numpy()
    res = {"laya_raw": sub.raw.to_numpy(float),
           "rules": sub.rule_uhs.fillna("none").map(RULE_SCORE).to_numpy(float)}
    def fill(vals):
        out = np.zeros(len(sub))
        out[reached] = vals
        return out
    if "laya_cal" in models:
        res["laya_cal"] = fill(M.apply_platt(sub.raw.to_numpy(float)[reached], models["laya_cal"]["a"], models["laya_cal"]["b"]))
    if "tfidf" in models and reached.any():
        res["tfidf"] = fill(models["tfidf"].predict_proba(tfidf_text(sub.title[reached], sub.abstract[reached]))[:, 1])
    if "stack" in models and reached.any():
        c = np.array(models["stack"]["coef"])
        X = stack_features(sub.raw.to_numpy(float)[reached], sub.rule_uhs.fillna("none")[reached])
        res["stack"] = fill(M.sigmoid(X @ c + models["stack"]["intercept"]))
    return res


def _oof(m, k, seed):
    strata = m.stratum.to_numpy()
    y = m.y.to_numpy()
    kk = int(min(k, max(2, min(int((y == 1).sum()), int((y == 0).sum())))))
    folds = M.stratified_folds(strata, y, kk, seed)
    oof = {}
    for f in range(kk):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        models = _fit_models(m, tr)
        sc = _apply(models, m, te)
        for name, v in sc.items():
            oof.setdefault(name, np.full(len(m), np.nan))[te] = v
    return oof


def _metrics_at(y, s, w, thr):
    c = M.wconf(y, (s >= thr).astype(int), w)
    return {**M.prf(c), **c}


def _sweep(y, s, w, name):
    rows = []
    for t in GRID:
        c = M.wconf(y, (s >= t).astype(int), w)
        p = M.prf(c)
        rows.append(dict(scorer=name, threshold=t, precision=p["precision"], recall=p["recall"], f1=p["f1"],
                         n_pred_pos=int((s >= t).sum())))
    return pd.DataFrame(rows)


def _r(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = None if np.isnan(v) else round(float(v), 3)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, tuple):
            out[k] = [None if (isinstance(x, float) and np.isnan(x)) else round(float(x), 3) for x in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(con, cfg, paths, log):
    vc = cfg["validation"]
    seed, target = int(vc["random_seed"]), float(vc["recall_target"])
    if not (paths.validation / SHEET).exists() or not (paths.validation / KEY).exists():
        raise FileNotFoundError("Run `python run.py sample` first and label the sheet.")
    m = gold_frame(con, cfg, paths, log)
    if len(m) < 30 or m.y.nunique() < 2:
        raise ValueError(f"Only {len(m)} labeled rows ({m.y.sum()} relevant): label at least ~100 (ideally 400) "
                         "with both classes present before evaluating.")
    y, w, strata = m.y.to_numpy(), m.w.to_numpy(float), m.stratum.to_numpy()
    out = {"n_labeled": int(len(m)), "n_relevant": int(y.sum()), "design": m.design.iloc[0],
           "strata": m.groupby("stratum").agg(population=("stratum_size", "first"), labeled=("y", "size"),
                                              relevant=("y", "sum")).reset_index().to_dict("records"),
           "model_tag": get_meta(con, "screen_tag"), "evaluated_at": now(), "recall_target": target}

    # human labels -> database (manual decisions override model decisions in `analyze`)
    con.executemany("INSERT OR REPLACE INTO manual VALUES(?,?,?,?,?)",
                    [(o, int(v), "gold", "", now()) for o, v in zip(m.openalex_id, m.y)])
    con.commit()

    oof = _oof(m, int(vc.get("cv_folds", 5)), seed)
    full = _fit_models(m, np.arange(len(m)))
    scorers = {"laya_raw": oof["laya_raw"], "rules": oof["rules"]}
    for k in ("laya_cal", "tfidf", "stack"):
        if k in oof and not np.isnan(oof[k]).any():
            scorers[k] = oof[k]
    table, sweeps = {}, []
    for name, s in scorers.items():
        sw = _sweep(y, s, w, name)
        sweeps.append(sw)
        ok = sw[sw.recall >= target]
        t_f1 = float(sw.loc[sw.f1.idxmax(), "threshold"]) if sw.f1.notna().any() else np.nan
        reached_target = len(ok) > 0
        t_rt = float(ok.threshold.max()) if reached_target else t_f1     # unreachable target -> best-F1 threshold
        if name == "rules":
            t_rt = t_f1 = 0.6
        mt = _metrics_at(y, s, w, t_rt)
        row = {"auc_w": M.auc_w(y, s, w), "recall_target_reachable": bool(reached_target),
               "max_recall_any_threshold": float(sw.recall.max()),
               "threshold_for_recall_target": t_rt, "threshold_max_f1": t_f1,
               "precision_at_recall_target": mt["precision"], "recall_at_recall_target": mt["recall"],
               "f1_at_recall_target": mt["f1"], "f1_max": float(sw.f1.max()) if sw.f1.notna().any() else np.nan}
        if name != "rules":
            row["ece_w"] = M.ece_w(y, s, w)
            row["brier_w"] = M.brier_w(y, s, w)
        table[name] = row
    out["scorers_cross_validated"] = {k: _r(v) for k, v in table.items()}
    # the raw rule used before validation, evaluated honestly
    thr_cfg = float(cfg["decision"]["raw_include_threshold"])
    out["laya_raw_at_config_threshold"] = _r({"threshold": thr_cfg, **_metrics_at(y, oof["laya_raw"], w, thr_cfg)})

    # ---------------------------------------------------------- operating point
    name = vc.get("operating_scorer", "laya_cal")
    if name not in scorers or name == "rules":
        log.warning("operating_scorer %r unavailable (needs enough labels / scikit-learn) - using laya_cal/raw", name)
        name = "laya_cal" if "laya_cal" in scorers else "laya_raw"
    thr = table[name]["threshold_for_recall_target"]
    if not table[name]["recall_target_reachable"]:
        log.warning("Recall target %.2f is NOT reachable with scorer %s (max recall %.3f; the gold sample contains UHS papers "
                    "that the pipeline cannot reach, e.g. in the auto-excluded stratum, or labels are noisy). Using the max-F1 "
                    "threshold %.2f instead - inspect validation_errors.csv.", target, name, table[name]["max_recall_any_threshold"], thr)
    s_op = scorers[name]
    reps = int(vc.get("bootstrap_reps", 1000))

    def stat(fn):
        return lambda idx: fn(M.prf(M.wconf(y[idx], (s_op[idx] >= thr).astype(int), w[idx])),
                              M.wconf(y[idx], (s_op[idx] >= thr).astype(int), w[idx]))
    mt = _metrics_at(y, s_op, w, thr)
    ci = {"precision": M.bootstrap(stat(lambda p, c: p["precision"]), strata, reps, seed),
          "recall": M.bootstrap(stat(lambda p, c: p["recall"]), strata, reps, seed),
          "f1": M.bootstrap(stat(lambda p, c: p["f1"]), strata, reps, seed),
          "estimated_missed": M.bootstrap(stat(lambda p, c: c["fn"]), strata, reps, seed)}
    out["operating_point"] = _r({"scorer": name, "threshold": thr, "precision": mt["precision"], "recall": mt["recall"],
                                 "f1": mt["f1"], "estimated_true_uhs_in_corpus": mt["tp"] + mt["fn"],
                                 "estimated_missed_uhs_papers": mt["fn"], "estimated_false_positives_in_included": mt["fp"],
                                 "precision_ci95": ci["precision"], "recall_ci95": ci["recall"], "f1_ci95": ci["f1"],
                                 "estimated_missed_ci95": ci["estimated_missed"]})
    op = {"scorer": name, "threshold": thr, "recall_target": target,
          "recall_target_reached": bool(table[name]["recall_target_reachable"]), "n_labeled": int(len(m)),
          "created_at": now(), "model_tag": out["model_tag"], "expected": out["operating_point"],
          "calibration": full.get("laya_cal")}
    if name == "stack":
        op["stack"] = full["stack"]
    if name == "tfidf":
        with open(paths.validation / "baseline_tfidf.pkl", "wb") as f:
            pickle.dump(full["tfidf"], f)
    if name == "laya_cal" and full.get("laya_cal"):
        a, b = full["laya_cal"]["a"], full["laya_cal"]["b"]
        raw_eq = float(M.sigmoid((M.logit(thr) - b) / a)) if a > 0 else None
        op["raw_equivalent_threshold"] = raw_eq
        gate = float(cfg["screening"]["stage2_min_prob"])
        if raw_eq is not None and raw_eq < gate:
            log.warning("The validated threshold corresponds to raw P=%.3f, BELOW screening.stage2_min_prob=%.2f: "
                        "lower stage2_min_prob (e.g. to %.2f) and re-run `screen` so every included paper is classified.",
                        raw_eq, gate, max(0.01, raw_eq - 0.05))
    (paths.validation / OP_FILE).write_text(json.dumps(op, indent=2, default=str), encoding="utf-8")
    if full.get("laya_cal"):
        (paths.validation / "calibration.json").write_text(json.dumps(full["laya_cal"], indent=2), encoding="utf-8")
    if "tfidf" in full and name != "tfidf":
        with open(paths.validation / "baseline_tfidf.pkl", "wb") as f:
            pickle.dump(full["tfidf"], f)

    # ------------------------------------------------- categorical / flags
    out["categorical"] = _categorical(m, cfg, paths)
    out["topic_flags"] = _flags(m, cfg)
    out["inter_rater"] = _inter_rater(m, paths)

    # --------------------------------------------------------------- outputs
    pd.concat(sweeps).to_csv(paths.validation / "threshold_sweep.csv", index=False)
    err = m.assign(score=s_op, error_type=np.where(y == 1, "FALSE_NEGATIVE", "FALSE_POSITIVE"))
    err = err[(err.y == 1) != (err.score >= thr)]
    err[["gold_id", "error_type", "score", "rule_uhs", "title", "abstract", "notes"]].to_csv(
        paths.validation / "validation_errors.csv", index=False, encoding="utf-8-sig")
    m.drop(columns=["abstract", "p_abstract", "answers_json"], errors="ignore").assign(oof_score=s_op).to_csv(
        paths.validation / "gold_merged.csv", index=False, encoding="utf-8-sig")
    (paths.validation / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    _plots(m, y, w, scorers, pd.concat(sweeps), name, thr, paths)
    set_meta(con, "validation_summary", {k: out[k] for k in ("n_labeled", "operating_point", "evaluated_at")})
    o = out["operating_point"]
    log.info("Validation: scorer=%s threshold=%.2f precision=%.3f recall=%.3f F1=%.3f | est. missed=%s (CI %s)",
             o["scorer"], o["threshold"], o["precision"], o["recall"], o["f1"], o["estimated_missed_uhs_papers"],
             o["estimated_missed_ci95"])
    return out


def _categorical(m, cfg, paths):
    min_conf = float(cfg["screening"]["min_choice_conf"])
    rel = m[m.y == 1]
    res = {}
    for hcol, (lcol, rcol, labels) in CATS.items():
        if hcol not in rel:
            continue
        sub = rel[rel[hcol].isin(labels)]
        if len(sub) < 10:
            continue
        lp = sub[lcol].fillna("not_classified")
        classified = lp != "not_classified"
        rp = sub[rcol].where(sub[rcol].isin(labels), None)
        fires = rp.notna()
        w = sub.w.to_numpy(float)
        h = sub[hcol].to_numpy()
        conf = sub[f"{lcol}_conf"]
        d = {"n": int(len(sub)), "laya_accuracy": float(np.average(lp.to_numpy() == h, weights=w)),
             "laya_macro_f1": M.macro_f1(h, lp.to_numpy(), labels, w), "laya_kappa": M.kappa(h, lp.to_numpy(), w),
             "laya_coverage": float(classified.mean()),
             "laya_accuracy_where_confident": float(np.average((lp.to_numpy() == h)[(conf >= min_conf).to_numpy()],
                                                               weights=w[(conf >= min_conf).to_numpy()])) if (conf >= min_conf).any() else None,
             "laya_accuracy_where_low_conf": float(np.average((lp.to_numpy() == h)[(conf < min_conf).to_numpy()],
                                                              weights=w[(conf < min_conf).to_numpy()])) if (conf < min_conf).any() else None,
             "rules_coverage": float(fires.mean()),
             "rules_accuracy_where_rule_fires": float(np.average((rp[fires].to_numpy() == h[fires.to_numpy()]),
                                                                 weights=w[fires.to_numpy()])) if fires.any() else None}
        res[hcol.replace("human_", "")] = _r(d)
        pd.crosstab(sub[hcol], lp, rownames=["human"], colnames=["laya"]).to_csv(paths.validation / f"confusion_{hcol}.csv")
    return res


def _flags(m, cfg):
    thr = float(cfg["screening"]["topic_threshold"])
    rel = m[m.y == 1]
    res = {}
    for hcol, q in FLAG_COLS.items():
        if hcol not in rel:
            continue
        sub = rel[pd.to_numeric(rel[hcol], errors="coerce").isin([0, 1])]
        if len(sub) < 20 or sub[hcol].nunique() < 2 or f"laya_{q}" not in sub:
            continue
        yy, pp, ww = sub[hcol].astype(int).to_numpy(), sub[f"laya_{q}"].to_numpy(float), sub.w.to_numpy(float)
        ok = ~np.isnan(pp)
        if ok.sum() < 20:
            continue
        c = M.wconf(yy[ok], (pp[ok] >= thr).astype(int), ww[ok])
        res[q] = _r({"n": int(ok.sum()), "auc_w": M.auc_w(yy[ok], pp[ok], ww[ok]), **M.prf(c)})
    return res


def _inter_rater(m, paths):
    f = paths.validation / SHEET2
    if not f.exists():
        return None
    g2 = pd.read_excel(f, sheet_name="Labels")
    g2["human_relevant"] = pd.to_numeric(g2["human_relevant"], errors="coerce")
    g1 = _read_labels(paths)
    j = g1.merge(g2[["gold_id", "human_relevant", "human_storage_type", "human_method", "human_scale"]],
                 on="gold_id", suffixes=("_1", "_2"))
    j = j.dropna(subset=["human_relevant_1", "human_relevant_2"])
    if len(j) < 10:
        return None
    out = {"n_double_labeled": int(len(j)),
           "agreement_relevance": float((j.human_relevant_1 == j.human_relevant_2).mean()),
           "kappa_relevance": M.kappa(j.human_relevant_1.astype(int), j.human_relevant_2.astype(int))}
    for c in ("human_storage_type", "human_method", "human_scale"):
        both = j[j[f"{c}_1"].notna() & j[f"{c}_2"].notna()]
        if len(both) >= 10:
            out[f"kappa_{c.replace('human_', '')}"] = M.kappa(both[f"{c}_1"].to_numpy(), both[f"{c}_2"].to_numpy())
    dis = j[j.human_relevant_1 != j.human_relevant_2]
    dis[["gold_id", "title", "abstract", "human_relevant_1", "human_relevant_2"]].assign(final_relevant="").to_csv(
        paths.validation / "rater_disagreements.csv", index=False, encoding="utf-8-sig")
    out["disagreements_file"] = "rater_disagreements.csv (fill final_relevant, save as adjudication.csv, re-run evaluate)"
    return _r(out)


def _plots(m, y, w, scorers, sweeps, name, thr, paths):
    sw = sweeps[sweeps.scorer == name]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(sw.threshold, sw.precision, label="Precision", color="#0072B2")
    ax.plot(sw.threshold, sw.recall, label="Recall", color="#D55E00")
    ax.plot(sw.threshold, sw.f1, label="F1", color="#009E73", lw=2.5)
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"deployed {thr:.2f}")
    ax.set_xlabel(f"Threshold on {name}"); ax.set_ylabel("Weighted score (cross-validated)"); ax.legend()
    ax.set_title("Threshold sweep against human gold labels", loc="left", fontweight="bold")
    fig.savefig(paths.validation / "val01_threshold_sweep.png", dpi=250, bbox_inches="tight"); plt.close(fig)

    s = scorers[name]
    c = M.wconf(y, (s >= thr).astype(int))
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(mat, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["excluded", "included"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["not UHS", "UHS"])
    ax.set_xlabel("Pipeline"); ax.set_ylabel("Human"); ax.set_title("Confusion (raw gold sample)", loc="left", fontweight="bold")
    fig.savefig(paths.validation / "val02_confusion.png", dpi=250, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    edges = np.linspace(0, 1, 11)
    for nm, col in (("laya_raw", "#D55E00"), ("laya_cal", "#0072B2")):
        if nm not in scorers:
            continue
        p = scorers[nm]
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            mk = idx == b
            if mk.sum() >= 3:
                xs.append(np.average(p[mk], weights=w[mk])); ys.append(np.average(y[mk], weights=w[mk]))
        ax.plot(xs, ys, "o-", color=col, label=nm.replace("_", " "))
    ax.set_xlabel("Predicted P(UHS)"); ax.set_ylabel("Observed share truly UHS (weighted)"); ax.legend()
    ax.set_title("Reliability (cross-validated)", loc="left", fontweight="bold")
    fig.savefig(paths.validation / "val03_calibration.png", dpi=250, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([s[y == 0], s[y == 1]], bins=25, weights=[w[y == 0], w[y == 1]], stacked=True, color=["#999", "#0072B2"],
            label=["human: not UHS", "human: UHS"])
    ax.axvline(thr, color="k", ls="--"); ax.set_xlabel(f"{name} score"); ax.set_ylabel("Population-weighted papers"); ax.legend()
    ax.set_title("Score distribution by human label", loc="left", fontweight="bold")
    fig.savefig(paths.validation / "val04_score_by_label.png", dpi=250, bbox_inches="tight"); plt.close(fig)


def _md_table(df):
    try:
        return df.to_markdown(index=False)
    except Exception:          # tabulate not installed
        return df.to_string(index=False)


# ------------------------------------------------------- compare Laya variants
def compare(con, cfg, paths, log, names=None, loader=None):
    """Screen ONLY the labeled gold papers with several Laya variants and rank them (no database writes)."""
    from .laya_compat import load_agent
    loader = loader or load_agent
    vc = cfg["validation"]
    target = float(vc["recall_target"])
    m = gold_frame(con, cfg, paths, log, need_laya=False)
    if len(m) < 30 or m.y.nunique() < 2:
        raise ValueError("Label at least ~100 gold papers (both classes) before `compare`.")
    variants = cfg.get("compare", {}).get("variants", {})
    names = names or list(variants)
    sc0 = cfg["screening"]
    y, w, strata = m.y.to_numpy(), m.w.to_numpy(float), m.stratum.to_numpy()
    reached = m.reached.to_numpy()
    rows = []
    for nm in names:
        v = variants[nm]
        sc = {**sc0, **{k: val for k, val in v.items() if k in ("checkpoint", "model_id", "question_style", "max_len", "head_max_len")}}
        log.info("compare: variant %s -> %s", nm, v)
        try:
            agent = loader(sc, log, checkpoint=sc["checkpoint"], style=sc["question_style"], max_len=sc.get("max_len"),
                           model_id=sc.get("model_id"), head_max_len=sc.get("head_max_len"))
        except Exception as e:
            log.error("variant %s could not be loaded: %s", nm, e)
            rows.append({"variant": nm, "error": f"{type(e).__name__}: {e}"})
            continue
        idx = np.where(reached)[0]
        p = np.zeros(len(m))
        trimmed = 0
        t0 = time.perf_counter()
        bs = int(sc0["batch_size"])
        for i in range(0, len(idx), bs):
            ch = idx[i:i + bs]
            comp = [compose_state(m.title.iloc[j], m.abstract.iloc[j], sc0["state_mode"], agent.text_budget,
                                  int(sc0["max_abstract_chars"])) for j in ch]
            trimmed += sum(c[1] for c in comp)
            res = agent.predict([c[0] for c in comp], Q_UHS, bs)
            for j, r in zip(ch, res):
                p[j] = r["answers"]["uhs"]["noul"]
        dt = time.perf_counter() - t0
        oof_cal = np.zeros(len(m))
        kk = int(min(int(vc.get("cv_folds", 5)), max(2, min(int((y == 1).sum()), int((y == 0).sum())))))
        folds = M.stratified_folds(strata, y, kk, int(vc["random_seed"]))
        for f in range(kk):
            tr = np.where((folds != f) & reached)[0]
            te = np.where((folds == f) & reached)[0]
            a, b = M.fit_platt(p[tr], y[tr], w[tr])
            oof_cal[te] = M.apply_platt(p[te], a, b)
        sw = _sweep(y, oof_cal, w, nm)
        ok = sw[sw.recall >= target]
        t = float(ok.threshold.max()) if len(ok) else float(sw.loc[sw.f1.idxmax(), "threshold"])
        mt = _metrics_at(y, oof_cal, w, t)
        rows.append({"variant": nm, "checkpoint": sc["checkpoint"], "question_style": sc["question_style"],
                     "max_len": agent.max_len, "auc_w": M.auc_w(y, p, w), "ece_raw": M.ece_w(y, p, w),
                     "ece_calibrated_cv": M.ece_w(y, oof_cal, w), f"precision_at_recall>={target}": mt["precision"],
                     "recall": mt["recall"], "threshold_cal": t, "share_p_in_0.4_0.6": float(((p > .4) & (p < .6))[reached].mean()),
                     "texts_trimmed_%": 100 * trimmed / max(len(idx), 1), "seconds_per_paper": dt / max(len(idx), 1),
                     "laya_version": agent.provenance().get("laya")})
    R = pd.DataFrame(rows)
    R.to_csv(paths.validation / "compare_summary.csv", index=False)
    col = f"precision_at_recall>={target}"
    good = R[R.get(col).notna()] if col in R else R.iloc[0:0]
    best = good.sort_values([col, "auc_w"], ascending=False).iloc[0] if len(good) else None
    md = ["# Laya variant comparison (gold labels, cross-validated calibration)", "", _md_table(R.round(3)), ""]
    if best is not None:
        md.append(f"**Best by precision at recall >= {target}: `{best['variant']}`** -> set in config.yaml: "
                  f"`screening.checkpoint: {best['checkpoint']}`, `screening.question_style: {best['question_style']}`"
                  + (f", `screening.max_len: {int(best['max_len'])}`" if best["checkpoint"] != "english" else "") + ".")
    (paths.validation / "compare_summary.md").write_text("\n".join(md), encoding="utf-8")
    log.info("compare: %s", "best = " + str(best["variant"]) if best is not None else "no variant succeeded")
    return R


# -------------------------------------------------------------- fine-tune export
def export_finetune(con, cfg, paths, log):
    """Gold labels -> JSONL in a generic state/questions/answers layout for fine-tuning a Laya checkpoint.

    The exact training-set format of the Laya SDK is defined by its fine-tuning notebook; adapt the few lines
    that read this file there (see fine_tune/README_FINETUNE.md). Splits are deterministic (hash of openalex_id).
    """
    from .questions import to_laya
    style = cfg["screening"].get("question_style", "choice_ab")
    m = gold_frame(con, cfg, paths, log, need_laya=False)
    qs = to_laya(QUESTIONS_ALL, style)
    out = paths.validation / "finetune_gold.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in m.itertuples():
            ans = {"uhs": "A" if r.y == 1 else "B"} if style == "choice_ab" else {"uhs": bool(r.y)}
            for hcol, (_, _, labs) in CATS.items():
                v = getattr(r, hcol, None)
                if r.y == 1 and isinstance(v, str) and v in labs:
                    ans[hcol.replace("human_", "")] = v
            h = int(hashlib.md5(str(r.openalex_id).encode()).hexdigest(), 16) % 10
            f.write(json.dumps({"gold_id": r.gold_id, "split": "test" if h == 0 else "dev" if h == 1 else "train",
                                "state": {"title": r.title, "abstract": r.abstract}, "questions": qs, "answers": ans,
                                "question_hash": question_hash(style)}, ensure_ascii=False) + "\n")
            n += 1
    log.info("Wrote %d fine-tuning records to %s", n, out)
    return out


# ----------------------------------------------------------- review ingestion
def ingest_review(con, path, log):
    """Read a filled review queue (csv/xlsx) with columns openalex_id, human_decision (1/0) into the manual table."""
    p = str(path)
    df = pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)
    df["human_decision"] = pd.to_numeric(df["human_decision"], errors="coerce")
    df = df.dropna(subset=["human_decision"])
    df = df[df.human_decision.isin([0, 1])]
    con.executemany("INSERT OR REPLACE INTO manual VALUES(?,?,?,?,?)",
                    [(o, int(d), "review", "", now()) for o, d in zip(df.openalex_id, df.human_decision)])
    con.commit()
    log.info("Ingested %d manual decisions from %s", len(df), p)
    return len(df)


# -------------------------------------------------------------- recall check
def _parse_seeds(text):
    seeds = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        title, doi = line, ""
        if "|" in line:
            title, doi = [x.strip() for x in line.split("|", 1)]
        elif line.lower().startswith("10."):
            title, doi = "", line
        seeds.append((title, norm_doi(doi)))
    return seeds


def recall_check(con, cfg, paths, log):
    f = paths.root / cfg["validation"]["seed_file"]
    if not f.exists():
        log.info("No seed file (%s) - skipping recall check", f)
        return None
    seeds = _parse_seeds(f.read_text(encoding="utf-8"))
    if not seeds:
        return None
    from .analyze import build_frame
    df = build_frame(con, cfg, paths)
    df = df[df.dup_of.isna()]
    ntitle = df.title.map(norm_title)
    rows = []
    for title, doi in seeds:
        hit = pd.DataFrame()
        if doi:
            hit = df[df.doi == doi]
        if hit.empty and title:
            nt = norm_title(title)
            exact = df[ntitle == nt]
            if exact.empty:
                sc = ntitle.map(lambda x: difflib.SequenceMatcher(None, nt, x).ratio() if abs(len(x) - len(nt)) < 25 else 0)
                exact = df[sc >= 0.93]
            hit = exact
        rows.append(dict(seed_title=title, seed_doi=doi, collected=len(hit) > 0,
                         decision=hit.decision.iloc[0] if len(hit) else "NOT COLLECTED",
                         score=hit.score_final.iloc[0] if len(hit) else np.nan,
                         matched_title=hit.title.iloc[0] if len(hit) else ""))
    R = pd.DataFrame(rows)
    R.to_csv(paths.validation / "seed_recall.csv", index=False, encoding="utf-8-sig")
    res = {"seeds": len(R), "collected": int(R.collected.sum()), "included": int(R.decision.isin(["included"]).sum()),
           "collection_recall": round(float(R.collected.mean()), 3),
           "pipeline_recall": round(float(R.decision.isin(["included"]).mean()), 3)}
    (paths.validation / "seed_recall.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("Seed recall: %s", res)
    return res
