"""Stage 4 - build the analysis frame (papers + Laya + rules + human decisions) and every summary table."""
from __future__ import annotations

import calendar
import datetime as dt
import json

import numpy as np
import pandas as pd

from .lexicon import GROUPS, TAGS
from .metrics import two_prop_p, wilson
from .questions import BINARY_DETAIL, CHOICE_DETAIL, Q_DETAIL
from .scorers import load_operating_point, score_frame

STORAGE_LABELS = {"depleted_gas_reservoir": "Depleted gas/HC reservoir", "aquifer": "Saline aquifer",
                  "salt_cavern": "Salt cavern", "hard_rock_cavern": "Hard-rock cavern / mine",
                  "other_geology": "Other geology (basalt, coal)", "general_or_unclear": "General / unclear",
                  "not_classified": "Not classified"}
METHOD_LABELS = {"experimental": "Experimental", "numerical": "Numerical", "analytical": "Analytical/theory",
                 "data_driven": "Data-driven / ML", "techno_economic": "Techno-economic / policy",
                 "review": "Review/perspective", "not_classified": "Not classified"}
SCALE_LABELS = {"pore": "Pore", "core": "Core / lab", "reservoir": "Reservoir", "field": "Field / regional",
                "other": "Unclear / other", "not_classified": "Not classified"}
NOUL_QS = list(BINARY_DETAIL)
CHOICE_QS = list(CHOICE_DETAIL)
PREPRINT_VENUES = ("ssrn", "arxiv", "earthrxiv", "eartharxiv", "essoar", "research square", "preprints.org",
                   "authorea", "chemrxiv", "engrxiv", "techrxiv")

COUNTRY = {"US": "United States", "CN": "China", "GB": "United Kingdom", "DE": "Germany", "AU": "Australia",
           "NO": "Norway", "IR": "Iran", "IN": "India", "CA": "Canada", "FR": "France", "NL": "Netherlands",
           "IT": "Italy", "ES": "Spain", "SA": "Saudi Arabia", "KR": "South Korea", "JP": "Japan", "BR": "Brazil",
           "AT": "Austria", "PL": "Poland", "CH": "Switzerland", "DK": "Denmark", "SE": "Sweden", "PT": "Portugal",
           "RU": "Russia", "AE": "UAE", "QA": "Qatar", "EG": "Egypt", "TR": "Turkey", "MY": "Malaysia",
           "PK": "Pakistan", "ID": "Indonesia", "SG": "Singapore", "ZA": "South Africa", "NG": "Nigeria",
           "BE": "Belgium", "FI": "Finland", "GR": "Greece", "IE": "Ireland", "HK": "Hong Kong", "CZ": "Czechia",
           "RO": "Romania", "HU": "Hungary", "UA": "Ukraine", "KZ": "Kazakhstan", "OM": "Oman", "KW": "Kuwait"}


def _loads(s):
    try:
        return json.loads(s) if isinstance(s, str) and s else {}
    except Exception:
        return {}


def _pmax(a):
    if not a or "choice" not in a:
        return np.nan
    return (a.get("probabilities") or {}).get(a["choice"], np.nan)


def decision_params(cfg, op):
    """(scorer name, threshold, half-width of the review band, validated?)."""
    hw = float(cfg["decision"]["review_halfwidth"])
    if op:
        return op["scorer"], float(op["threshold"]), hw, True
    return "laya_raw", float(cfg["decision"]["raw_include_threshold"]), hw, False


def build_frame(con, cfg, paths=None, log=None) -> pd.DataFrame:
    sc = cfg["screening"]
    op = load_operating_point(paths) if (paths is not None and cfg["decision"].get("use_validated_operating_point", True)) else None
    p = pd.read_sql_query("SELECT * FROM papers", con)
    l = pd.read_sql_query("SELECT openalex_id,stage,uhs_prob,answers_json,tokens_per_row,near_max_len,"
                          "model_tag,screened_at,est_tokens,trimmed FROM laya", con)
    man = pd.read_sql_query("SELECT openalex_id,decision AS manual_decision,source AS manual_source FROM manual", con)
    df = p.merge(l, on="openalex_id", how="left").merge(man, on="openalex_id", how="left")
    A = df["answers_json"].map(_loads)
    for q in NOUL_QS:
        df[f"laya_{q}"] = A.map(lambda a, q=q: (a.get(q) or {}).get("noul", np.nan))
    for q in CHOICE_QS:
        df[f"laya_{q}"] = A.map(lambda a, q=q: (a.get(q) or {}).get("choice"))
        df[f"laya_{q}_conf"] = A.map(lambda a, q=q: _pmax(a.get(q)))
    df = df.drop(columns=["answers_json"])

    scorer, thr, hw, validated = decision_params(cfg, op)
    df["score_final"], used = score_frame(df, op, paths, log)
    if used != scorer:
        scorer, validated = used, False
    df.attrs.update(scorer=scorer, threshold=thr, halfwidth=hw, validated=validated, op=op)
    lo, hi = max(0.0, thr - hw), min(1.0, thr + hw)

    has_s = df["score_final"].notna()
    model_dec = np.select(
        [df.dup_of.notna(), df.eligibility.ne("eligible"), ~has_s, df.score_final >= thr],
        ["duplicate", "auto_excluded", "pending", "included"], default="excluded")
    df["model_decision"] = model_dec
    man_d = df["manual_decision"]
    human = man_d.notna() & df.dup_of.isna()
    dec = pd.Series(model_dec, index=df.index)
    dec[human & (man_d == 1)] = "included"
    dec[human & (man_d == 0) & df.eligibility.eq("eligible")] = "excluded"   # (auto-excluded records stay so)
    df["decision"] = dec
    df["decision_source"] = np.where(human, "human", "model")

    screened = df.decision.isin(["included", "excluded"]) & df.score_final.notna() & df.decision_source.eq("model")
    s_pos = df.score_final >= thr
    rule_pos = df.rule_uhs.isin(["strong", "likely"])
    df["abstract_missing"] = df.abstract.fillna("").str.len().lt(50)
    df["review_priority"] = np.select(
        [screened & ~s_pos & df.rule_uhs.eq("strong"),
         screened & s_pos & ~rule_pos,
         screened & ~s_pos & df.rule_uhs.eq("likely"),
         screened & df.abstract_missing & (rule_pos | df.score_final.between(lo, hi)),
         screened & df.score_final.between(lo, hi)],
        ["A_possible_false_negative", "B_possible_false_positive", "C_rules_disagree", "E_title_only_check", "D_borderline"],
        default="")
    df["confidence_tier"] = np.select(
        [~df.decision.isin(["included", "excluded"]), df.decision_source.eq("human"), df.score_final >= hi,
         df.score_final >= thr, df.score_final >= lo],
        ["", "human_decision", "high_include", "include", "borderline_exclude"], default="clear_exclude")

    mc = float(sc["min_choice_conf"])
    for q in CHOICE_QS:
        df[f"{q}_lowconf"] = df[f"laya_{q}_conf"].lt(mc)
    tsets = df["tags"].fillna("").map(lambda s: {x for x in s.split(";") if x})
    for t in TAGS:
        df[f"tag_{t}"] = tsets.map(lambda s, t=t: t in s)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
    df["pub_month"] = df["publication_date"].fillna("").astype(str).str[:7]
    v = df["journal"].fillna("").str.lower()
    df["is_preprint"] = df["type"].eq("preprint") | v.map(lambda x: any(k in x for k in PREPRINT_VENUES))
    return df


def prisma_counts(con, df, cfg) -> dict:
    thr, hw = df.attrs["threshold"], df.attrs["halfwidth"]
    lo, hi = max(0.0, thr - hw), min(1.0, thr + hw)
    hits = con.execute("SELECT COUNT(*) FROM paper_queries").fetchone()[0]
    inc, exc = df.decision.eq("included"), df.decision.eq("excluded")
    auto = df[df.decision.eq("auto_excluded")]
    hum = df.decision_source.eq("human")
    return {
        "query_hits_total_with_overlap": int(hits), "unique_records": int(len(df)),
        "duplicates_removed": int(df.decision.eq("duplicate").sum()),
        "unique_after_dedup": int((~df.decision.eq("duplicate")).sum()),
        "auto_excluded": int(len(auto)), "auto_excluded_reasons": auto.exclude_reason.value_counts().to_dict(),
        "eligible_for_screening": int(df.decision.isin(["included", "excluded", "pending"]).sum()),
        "screened_by_laya": int((df.score_final.notna() & df.dup_of.isna() & df.eligibility.eq("eligible")).sum()),
        "pending_screening": int(df.decision.eq("pending").sum()),
        "model_included": int(df.model_decision.eq("included").sum()),
        "model_excluded": int(df.model_decision.eq("excluded").sum()),
        "excluded_by_laya": int((exc & ~hum).sum()), "included": int(inc.sum()),
        "included_by_model": int((inc & ~hum).sum()),
        "human_included": int((inc & hum).sum()), "human_excluded": int((exc & hum).sum()),
        "human_added": int((inc & df.model_decision.ne("included")).sum()),
        "human_removed": int((exc & df.model_decision.eq("included")).sum()),
        "included_high_confidence": int((inc & ~hum & (df.score_final >= hi)).sum()),
        "included_borderline": int((inc & ~hum & (df.score_final < hi)).sum()),
        "excluded_borderline": int((exc & ~hum & (df.score_final >= lo)).sum()),
        "included_title_only": int((inc & df.abstract_missing).sum()),
        "review_queue_size": int(df.review_priority.ne("").sum()),
        "scorer": df.attrs["scorer"], "threshold": round(float(thr), 3), "validated": bool(df.attrs["validated"]),
        "review_band": [round(lo, 3), round(hi, 3)],
    }


def _count(series, top=None, sep=None):
    s = series.dropna()
    if sep:
        s = s.astype(str).str.split(sep).explode().str.strip()
    s = s[s.astype(str).str.len() > 0]
    vc = s.value_counts()
    return vc.head(top) if top else vc


def _months_covered(cfg, year):
    start, end = [dt.date.fromisoformat(x) for x in cfg["_window"]]
    y0, y1 = start.year, end.year
    if year < y1 and year > y0:
        return 12.0
    lo = start if year == y0 else dt.date(year, 1, 1)
    hi = end if year == y1 else dt.date(year, 12, 31)
    def pos(d):
        return d.month - 1 + d.day / calendar.monthrange(d.year, d.month)[1]
    return round(pos(hi) - (lo.month - 1 + (lo.day - 1) / calendar.monthrange(lo.year, lo.month)[1]), 2)


def make_tables(con, df, cfg) -> dict:
    sc, top = cfg["screening"], int(cfg["analysis"]["top_n"])
    thr = df.attrs["threshold"]
    T = {}
    inc = df[df.decision == "included"].copy()
    for q, labels in (("storage_type", STORAGE_LABELS), ("method", METHOD_LABELS), ("scale", SCALE_LABELS)):
        inc[f"{q}_lab"] = inc[f"laya_{q}"].fillna("not_classified").map(lambda v, L=labels: L.get(v, v))

    base = df[df.decision != "duplicate"]
    y0_, y1_ = cfg["_start_year"], cfg["_end_year"]
    inw = base[base.publication_year.between(y0_, y1_)]          # trends only inside the publication window
    y = inw.groupby("publication_year").decision.value_counts().unstack(fill_value=0)
    y.columns.name = None
    y.index.name = "publication_year"
    for c in ("included", "excluded", "auto_excluded", "pending"):
        if c not in y:
            y[c] = 0
    y["records"] = y.sum(axis=1)
    y["include_rate_%"] = (100 * y.included / (y.included + y.excluded).replace(0, np.nan)).round(1)
    y["months_covered"] = [_months_covered(cfg, int(i)) for i in y.index]
    y["partial_year"] = y.months_covered < 11.99
    y["included_per_month"] = (y.included / y.months_covered).round(1)
    y["included_per_month_change_%"] = (100 * y.included_per_month.pct_change()).round(1)
    T["yearly"] = y[["records", "auto_excluded", "excluded", "included", "pending", "include_rate_%", "months_covered",
                     "partial_year", "included_per_month", "included_per_month_change_%"]].reset_index()

    mo = inw[inw.pub_month.str.len() == 7].groupby("pub_month").decision.value_counts().unstack(fill_value=0)
    mo.columns.name = None
    for c in ("included", "excluded", "auto_excluded", "pending"):
        if c not in mo:
            mo[c] = 0
    mo["records"] = mo.sum(axis=1)
    T["monthly"] = mo[["records", "included", "excluded", "auto_excluded", "pending"]].reset_index()
    pp = inc.assign(is_preprint=inc.is_preprint).groupby("publication_year").is_preprint.agg(["sum", "size"])
    T["preprint_share"] = pp.rename(columns={"sum": "preprints", "size": "included"}).assign(
        share_pct=lambda d: (100 * d.preprints / d["included"]).round(1)).reset_index()

    for q in ("storage_type", "method", "scale"):
        T[f"{q}_counts"] = inc[f"{q}_lab"].value_counts().rename_axis(q).reset_index(name="n").assign(
            pct=lambda d: (100 * d.n / d.n.sum()).round(1))
        T[f"{q}_by_year"] = pd.crosstab(inc.publication_year, inc[f"{q}_lab"]).reset_index()
    T["storage_x_method"] = pd.crosstab(inc.storage_type_lab, inc.method_lab).reset_index()
    T["storage_x_scale"] = pd.crosstab(inc.storage_type_lab, inc.scale_lab).reset_index()
    T["method_x_scale"] = pd.crosstab(inc.method_lab, inc.scale_lab).reset_index()

    # ---- topics: Laya flags and lexicon tags
    rows = []
    for q in NOUL_QS:
        v = inc[f"laya_{q}"]
        k = int((v >= sc["topic_threshold"]).sum())
        rows.append(dict(source="Laya", topic=q, group="laya_question", n=k,
                         pct=round(100 * k / max(len(inc), 1), 1)))
    for t in TAGS:
        n = int(inc[f"tag_{t}"].sum())
        rows.append(dict(source="Lexicon", topic=t, group=GROUPS[t], n=n, pct=round(100 * n / max(len(inc), 1), 1)))
    T["topic_prevalence"] = pd.DataFrame(rows).sort_values(["source", "n"], ascending=[True, False])
    tagcols = [f"tag_{t}" for t in TAGS]
    tby = inc.groupby("publication_year")[tagcols].mean().mul(100).round(1)
    tby.columns = [c[4:] for c in tagcols]
    T["tag_pct_by_year"] = tby.reset_index()

    # ---- topic shift first -> last year (shares are comparable even when the last year is partial)
    yrs = sorted(int(v) for v in inc.publication_year.dropna().unique())
    if len(yrs) >= 2:
        a, b = yrs[0], yrs[-1]
        ia, ib = inc[inc.publication_year == a], inc[inc.publication_year == b]
        sh = []
        for t in TAGS:
            ka, kb = int(ia[f"tag_{t}"].sum()), int(ib[f"tag_{t}"].sum())
            sh.append(dict(topic=t, group=GROUPS[t], **{f"pct_{a}": round(100 * ka / max(len(ia), 1), 1),
                                                       f"pct_{b}": round(100 * kb / max(len(ib), 1), 1)},
                           diff_pp=round(100 * (kb / max(len(ib), 1) - ka / max(len(ia), 1)), 1),
                           p_value=round(two_prop_p(ka, len(ia), kb, len(ib)), 4) if len(ia) and len(ib) else np.nan))
        sh = pd.DataFrame(sh)
        sh["p_adj_bonferroni"] = (sh.p_value * len(sh)).clip(upper=1).round(4)
        T["topic_shift"] = sh.sort_values("diff_pp", ascending=False)

    # ---- research-coverage matrix (setting x process) + gaps
    settings = [t for t in TAGS if GROUPS[t] == "setting"]
    topics = [t for t in TAGS if GROUPS[t] in ("flow_physics", "coupled_process", "operations")]
    cov = pd.DataFrame({tp: {s: int((inc[f"tag_{s}"] & inc[f"tag_{tp}"]).sum()) for s in settings} for tp in topics})
    T["coverage_matrix"] = cov.rename_axis("setting").reset_index()
    gaps = cov.stack().rename("n_papers").reset_index()
    gaps.columns = ["setting", "topic", "n_papers"]
    gaps["setting_total"] = gaps.setting.map(lambda s: int(inc[f"tag_{s}"].sum()))
    gaps["topic_total"] = gaps.topic.map(lambda t: int(inc[f"tag_{t}"].sum()))
    N = max(len(inc), 1)
    gaps["expected_if_independent"] = (gaps.setting_total * gaps.topic_total / N).round(1)
    gaps["lift"] = (gaps.n_papers / gaps.expected_if_independent.replace(0, np.nan)).round(2)
    thr_g = int(cfg["analysis"]["gap_threshold"])
    mn = int(cfg["analysis"]["gap_min_setting_papers"])
    lift_below = float(cfg["analysis"]["gap_lift_below"])
    gaps["gap_type"] = np.select(
        [gaps.n_papers <= thr_g,
         (gaps.setting_total >= mn) & (gaps.topic_total >= mn) & (gaps.expected_if_independent >= 5) & (gaps.lift < lift_below)],
        [f"absolute (<= {thr_g} papers)", f"relative (lift < {lift_below})"], default="")
    T["coverage_gaps"] = gaps[gaps.gap_type != ""].sort_values(["gap_type", "n_papers", "setting"])

    # ---- co-occurrence of top tags (non-setting)
    cand = [t for t in TAGS if GROUPS[t] not in ("setting",)]
    topn = sorted(cand, key=lambda t: -inc[f"tag_{t}"].sum())[:20]
    Mx = inc[[f"tag_{t}" for t in topn]].astype(int)
    co = Mx.T.dot(Mx)
    co.index = co.columns = topn
    T["tag_cooccurrence"] = co.rename_axis("tag").reset_index()

    # ---- bibliometrics (whole + fractional country counting)
    T["top_journals"] = _count(inc.journal, 25).rename_axis("journal").reset_index(name="n")
    cc = _count(inc.countries, 25, ";")
    T["top_countries"] = cc.rename_axis("code").reset_index(name="n").assign(
        country=lambda d: d.code.map(lambda c: COUNTRY.get(c, c)))[["code", "country", "n"]]
    frac = {}
    for s in inc.countries.fillna(""):
        cs = [c for c in s.split(";") if c]
        for c in cs:
            frac[c] = frac.get(c, 0) + 1 / len(cs)
    fr = pd.Series(frac).sort_values(ascending=False).head(25).round(1)
    T["top_countries_fractional"] = fr.rename_axis("code").reset_index(name="n_fractional").assign(
        country=lambda d: d.code.map(lambda c: COUNTRY.get(c, c)))[["code", "country", "n_fractional"]]
    ncs = inc.countries.fillna("").map(lambda s: len([c for c in s.split(";") if c]))
    T["collaboration"] = pd.DataFrame([{"papers_with_country_data": int((ncs > 0).sum()),
                                        "international_collab_%": round(100 * float((ncs > 1).sum() / max((ncs > 0).sum(), 1)), 1)}])
    T["top_institutions"] = _count(inc.institutions, 25, ";").rename_axis("institution").reset_index(name="n")
    T["top_keywords"] = _count(inc.keywords, 40, ";").rename_axis("keyword").reset_index(name="n")
    T["top_tools"] = _count(inc.tools, None, ";").rename_axis("tool").reset_index(name="n")
    T["oa_status"] = inc.oa_status.replace("", "unknown").value_counts().rename_axis("oa_status").reset_index(name="n")
    T["top_cited"] = inc.sort_values("cited_by_count", ascending=False).head(30)[
        ["cited_by_count", "publication_year", "title", "journal", "doi", "is_preprint", "storage_type_lab", "method_lab", "score_final"]]
    T["top_publishers"] = _count(inc.publisher, 15).rename_axis("publisher").reset_index(name="n")

    # ---- threshold sensitivity (how fragile is the corpus size?)
    sc_df = df[df.decision.isin(["included", "excluded"]) & df.decision_source.eq("model")]
    grid = np.round(np.arange(0.10, 0.951, 0.05), 2)
    base_n = int((sc_df.score_final >= thr).sum())
    ts = pd.DataFrame({"threshold": grid, "included_by_model": [int((sc_df.score_final >= g).sum()) for g in grid]})
    ts["delta_vs_current"] = ts.included_by_model - base_n
    ts["rule_strong_or_likely_among_included_%"] = [
        round(100 * float(sc_df[sc_df.score_final >= g].rule_uhs.isin(["strong", "likely"]).mean()), 1)
        if (sc_df.score_final >= g).any() else np.nan for g in grid]
    T["threshold_sensitivity"] = ts

    # ---- Laya vs rule agreement
    T["agreement_uhs"] = pd.crosstab(sc_df.rule_uhs, np.where(sc_df.score_final >= thr, "laya_include", "laya_exclude")).reset_index()
    T["agreement_storage"] = pd.crosstab(inc.laya_storage_type.fillna("nc"), inc.rule_storage).reset_index()
    rq = df[df.review_priority != ""].assign(_dist=lambda d: (d.score_final - thr).abs()).sort_values(
        ["review_priority", "_dist"]).drop(columns="_dist")
    rq = rq[["review_priority", "openalex_id", "doi", "publication_year", "title", "abstract", "score_final",
             "uhs_prob", "rule_uhs", "laya_storage_type", "rule_storage", "abstract_missing"]]
    rq["human_decision"] = ""          # fill 1 (UHS) / 0 (not UHS), then `python run.py ingest-review <file>`
    T["review_queue"] = rq

    # ---- query yield
    hits = pd.read_sql_query("SELECT query_id, COUNT(*) hits FROM paper_queries GROUP BY query_id", con)
    pq = pd.read_sql_query("SELECT * FROM paper_queries", con)
    nq = pq.groupby("openalex_id").size().rename("n_queries")
    pq = pq.merge(nq, on="openalex_id").merge(df[["openalex_id", "decision"]], on="openalex_id")
    incq = pq[pq.decision == "included"].groupby("query_id").agg(
        included=("openalex_id", "size"), unique_included=("n_queries", lambda s: int((s == 1).sum())))
    qy = hits.merge(incq, on="query_id", how="left").fillna(0)
    qy["included"], qy["unique_included"] = qy.included.astype(int), qy.unique_included.astype(int)
    qy["precision_%"] = (100 * qy.included / qy.hits).round(1)
    sl = pd.read_sql_query("SELECT query_id, query, api_count, MAX(run_at) last_run, status FROM search_log GROUP BY query_id", con)
    T["query_yield"] = sl.merge(qy, on="query_id", how="outer")

    # ---- data quality / sanity checks
    nonlow = inc.storage_type_lowconf.fillna(False) if len(inc) else pd.Series(dtype=bool)
    dq = {
        "records_unique": int((df.decision != "duplicate").sum()),
        "abstract_missing_%": round(100 * float(base.abstract_missing.mean()), 1),
        "abstract_source_openalex_%": round(100 * float((base.abstract_source == "openalex").mean()), 1),
        "abstract_source_crossref_%": round(100 * float((base.abstract_source == "crossref").mean()), 1),
        "abstract_source_semanticscholar_%": round(100 * float((base.abstract_source == "semanticscholar").mean()), 1),
        "doi_missing_%": round(100 * float((base.doi.fillna("") == "").mean()), 1),
        "countries_missing_%_included": round(100 * float((inc.countries.fillna("") == "").mean()), 1) if len(inc) else 0,
        "included_screened_on_title_only": int(inc.abstract_missing.sum()),
        "texts_trimmed_to_context_budget_included": int(inc.trimmed.fillna(0).sum()),
        "laya_inputs_at_max_len": int(df.near_max_len.fillna(0).sum()),
        "storage_low_confidence_included": int(nonlow.sum()) if len(inc) else 0,
        "included_not_stage2_classified": int((inc.laya_storage_type.isna() & inc.decision_source.eq("model")).sum()) if len(inc) else 0,
        "human_decisions_applied": int(df.decision_source.eq("human").sum()),
        "sanity_auto_excluded_but_score_high": int(((df.decision == "auto_excluded") & (df.score_final >= thr)).sum()),
        "distinct_model_tags": int(df.model_tag.nunique()),
    }
    T["data_quality"] = pd.DataFrame(list(dq.items()), columns=["check", "value"])
    return T
