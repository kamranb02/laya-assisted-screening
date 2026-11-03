"""Stage 7 - single-file HTML report (figures embedded, print-to-PDF friendly)."""
from __future__ import annotations

import base64
import html
import json

import pandas as pd

from . import __version__
from .common import get_meta

CSS = """
body{font-family:Arial,Helvetica,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#1b1f24;line-height:1.5}
h1{color:#1F4E79;border-bottom:3px solid #1F4E79;padding-bottom:.3rem}h2{color:#1F4E79;margin-top:2.2rem;border-bottom:1px solid #ccd}
h3{margin-bottom:.2rem}.cards{display:flex;flex-wrap:wrap;gap:.8rem}.card{flex:1 1 150px;background:#F3F7FB;border-left:5px solid #0072B2;padding:.6rem .9rem}
.card b{font-size:1.6rem;display:block}.warn{background:#FFF4E0;border-left:5px solid #E69F00;padding:.7rem 1rem;margin:1rem 0}
.ok{background:#E8F6EE;border-left:5px solid #009E73;padding:.7rem 1rem;margin:1rem 0}
table{border-collapse:collapse;font-size:.85rem;margin:.6rem 0;width:100%}th{background:#1F4E79;color:#fff;text-align:left}
td,th{border:1px solid #d5dbe3;padding:4px 7px;vertical-align:top}tr:nth-child(even){background:#f7f9fb}
figure{margin:1.2rem 0}figcaption{font-size:.85rem;color:#444}img{max-width:100%}code{background:#eef;padding:1px 4px}
@media print{.pb{page-break-before:always}}
"""


def _img(paths, name, folder=None):
    f = (folder or paths.figures) / f"{name}.png"
    if not f.exists():
        return ""
    b = base64.b64encode(f.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{name}">'


def _fig(paths, name, cap):
    i = _img(paths, name)
    return f"<figure>{i}<figcaption><b>Figure.</b> {html.escape(cap)}</figcaption></figure>" if i else ""


def _tbl(df, n=None, cols=None):
    if df is None or len(df) == 0:
        return "<p><i>No data.</i></p>"
    d = df.head(n) if n else df
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    return d.to_html(index=False, escape=True, border=0, float_format=lambda v: f"{v:.3g}", na_rep="")


def _load(f):
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _kv(d, keys=None):
    rows = [(k, json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in (d or {}).items() if not keys or k in keys]
    return _tbl(pd.DataFrame(rows, columns=["item", "value"]))


def build_report(df, T, P, cfg, paths, log, con=None):
    sc = cfg["screening"]
    inc = df[df.decision == "included"]
    val = _load(paths.validation / "metrics.json")
    smoke = _load(paths.validation / "smoke_report.json")
    seed = _load(paths.validation / "seed_recall.json")
    Y = T["yearly"]
    start, end = cfg["_window"]
    yr = {int(r.publication_year): (int(r.included), float(r.included_per_month)) for r in Y.itertuples()}
    y0, y1 = min(yr), max(yr)
    growth = f"{yr[y1][1] / yr[y0][1]:.2f}×" if yr[y0][1] else "n/a"
    top_storage = T["storage_type_counts"].iloc[0] if len(T["storage_type_counts"]) else None
    dq = dict(zip(T["data_quality"].check, T["data_quality"].value))
    prov = (get_meta(con, "screen_provenance") if con is not None else None) or {}
    tag = (get_meta(con, "screen_tag") if con is not None else None) or "n/a"

    H = [f"<html><head><meta charset='utf-8'><title>{html.escape(cfg['project']['name'])}</title><style>{CSS}</style></head><body>",
         f"<h1>{html.escape(cfg['project']['name'])}</h1>",
         f"<p>Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} · pipeline v{__version__} · config hash <code>{cfg['_hash']}</code> · "
         f"window {start} … {end} · decision score <code>{P['scorer']}</code> ≥ {P['threshold']}</p>"]

    if P["validated"] and val:
        o = val["operating_point"]
        H.append(f"<div class='ok'><b>Validated against {val['n_labeled']} human-labeled papers</b> (design: {val['design']}). "
                 f"Deployed scorer <code>{o['scorer']}</code> at threshold {o['threshold']} (chosen for recall ≥ {val['recall_target']}): "
                 f"precision {o['precision']} (95% CI {o['precision_ci95']}), recall {o['recall']} ({o['recall_ci95']}), F1 {o['f1']}. "
                 f"Estimated UHS papers in the collected records: ≈{o['estimated_true_uhs_in_corpus']:,}; estimated missed: "
                 f"≈{o['estimated_missed_uhs_papers']:,} (95% CI {o['estimated_missed_ci95']}). Cross-validated estimates.</div>")
    else:
        H.append("<div class='warn'><b>Classification not yet validated.</b> All model numbers below are raw outputs, not ground truth. "
                 "Run <code>python run.py sample</code>, label the sheet, then <code>python run.py evaluate</code> and re-generate this report.</div>")
    if smoke and not smoke.get("overall_pass", True):
        H.append("<div class='warn'><b>Laya smoke test FAILED</b> on this machine - see validation/smoke_report.md.</div>")
    if P["pending_screening"]:
        H.append(f"<div class='warn'><b>{P['pending_screening']:,} eligible records are still pending Laya screening</b>; counts are incomplete.</div>")
    if dq.get("distinct_model_tags", 1) > 1:
        H.append("<div class='warn'><b>Several Laya configurations are mixed in the database</b> (distinct_model_tags &gt; 1).</div>")

    cards = [(f"{P['unique_after_dedup']:,}", "unique records"), (f"{P['included']:,}", "included UHS papers"),
             (f"{P['review_queue_size']:,}", "in manual-review queue"), (growth, f"included/month {y0}→{y1}*"),
             (f"{dq.get('abstract_missing_%', 0)}%", "records without abstract")]
    H.append("<div class='cards'>" + "".join(f"<div class='card'><b>{a}</b>{html.escape(b)}</div>" for a, b in cards) + "</div>")
    H.append(f"<p class='small'>* {y1} is a partial year (data to {end}); growth compares <b>papers per month</b>.</p>")

    H.append("<h2>1. Method in brief</h2><ol>"
             f"<li><b>Collection</b>: {len(cfg['queries'])} Boolean queries on OpenAlex (title+abstract), {start} … {end}; "
             "cursor pagination without result caps; every query hit stored.</li>"
             "<li><b>Abstract recovery</b>: missing abstracts are filled from Crossref and Semantic Scholar before screening.</li>"
             "<li><b>De-duplication</b>: same DOI or same normalised title (journal version preferred over preprint).</li>"
             "<li><b>Rule pre-filter</b>: errata/retractions, out-of-range dates and records with no hydrogen term in title+abstract are excluded automatically.</li>"
             f"<li><b>Laya screening</b>: stage 1 asks whether the paper is about UHS; stage 2 (raw P ≥ {sc['stage2_min_prob']}) classifies storage setting, method, scale and topic flags.</li>"
             "<li><b>Independent keyword rules</b> add ~50 reservoir-engineering tags and a second opinion; disagreements form the review queue.</li>"
             "<li><b>Validation</b>: blinded human gold sample (rule-stratified), cross-validated calibration, baselines, bootstrap CIs, threshold chosen for the recall target (§8).</li>"
             "<li><b>Human-in-the-loop</b>: gold labels and review-queue decisions override the model (decision_source = human).</li></ol>")
    H.append("<h3>Model and settings used (provenance)</h3>" + _kv({"model_tag": tag, **prov, "question_style": sc["question_style"],
                                                                      "state_mode": sc["state_mode"], "batch_size": sc["batch_size"]}))

    H.append("<h2 class='pb'>2. Record flow</h2>" + _fig(paths, "fig01_prisma_flow", "PRISMA-style flow of records.") +
             f"<p>Auto-exclusions: {html.escape(json.dumps(P['auto_excluded_reasons']))}.</p>")

    H.append("<h2 class='pb'>3. Publication trend</h2>" + _fig(paths, "fig02_publications_per_year", "Eligible vs included papers per (partial) year with the per-month rate.") +
             _fig(paths, "fig24_monthly", "Monthly publications. The most recent months are under-indexed.") + _tbl(Y) +
             "<h3>Preprints among included papers</h3>" + _tbl(T["preprint_share"]))
    if top_storage is not None:
        H.append(f"<h2>4. Storage setting, method and scale</h2><p>The most frequent setting is <b>{html.escape(str(top_storage.storage_type))}</b> "
                 f"({int(top_storage.n)} papers, {top_storage.pct}%). Categorical answers are only as good as the validation in §8.</p>")
    for n, c in (("fig05_storage_setting", "Storage setting."), ("fig05b_storage_by_year", "Storage setting by year."),
                 ("fig06_method", "Research method."), ("fig06b_method_by_year", "Method by year."), ("fig07_scale", "Scale."),
                 ("fig08_storage_x_method", "Setting × method."), ("fig09_storage_x_scale", "Setting × scale.")):
        H.append(_fig(paths, n, c))

    H.append("<h2 class='pb'>5. Reservoir-engineering & porous-media coverage</h2>"
             "<p>Tags are abstract-level mentions (multi-label); they mean 'mentioned', not 'main topic'. The coverage matrix shows where the literature is dense and where it is thin.</p>")
    for n, c in (("fig10_topic_prevalence", "Most frequent topics."), ("fig11_topic_trends", "Topic trends by year."),
                 ("fig25_topic_shift", "Topic shift between the first and last year."),
                 ("fig12_coverage_matrix", "Setting × process coverage matrix (paper counts)."),
                 ("fig13_tag_cooccurrence", "Topic co-occurrence."), ("fig18_lithology_fluids", "Lithology and fluids."),
                 ("fig19_simulation_tools", "Software tools.")):
        H.append(_fig(paths, n, c))
    H.append(f"<h3>Potential research gaps (absolute ≤ {cfg['analysis']['gap_threshold']} papers, or relative lift &lt; {cfg['analysis']['gap_lift_below']})</h3>"
             + _tbl(T["coverage_gaps"], 60))
    H.append("<h3>Topic shift (first → last year)</h3>" + _tbl(T.get("topic_shift"), 25))

    H.append("<h2 class='pb'>6. Bibliometrics</h2>")
    for n, c in (("fig14_top_journals", "Top journals."), ("fig15_top_countries", "Countries (whole counting)."),
                 ("fig16_top_keywords", "Keywords."), ("fig17_citations_oa", "Citations and open access.")):
        H.append(_fig(paths, n, c))
    H.append("<h3>Countries, fractional counting</h3>" + _tbl(T["top_countries_fractional"], 15) +
             "<h3>International collaboration</h3>" + _tbl(T["collaboration"]) +
             "<h3>Most cited included papers</h3><p class='small'>Citation counts of 2025–2026 papers are small and strongly age-biased.</p>" + _tbl(T["top_cited"], 20))

    H.append("<h2 class='pb'>7. Classification behaviour</h2>" + _fig(paths, "fig03_relevance_distribution", "Distribution of the decision score.") +
             _fig(paths, "fig04_threshold_sensitivity", "Corpus size vs threshold.") + _fig(paths, "fig20_laya_vs_rules", "Model decision vs keyword-rule class.") +
             _fig(paths, "fig21_laya_confidence", "Confidence of Laya categorical answers."))

    H.append("<h2>8. Validation against human labels</h2>")
    if val:
        sc_tab = pd.DataFrame(val["scorers_cross_validated"]).T.reset_index().rename(columns={"index": "scorer"})
        H.append("<h3>Scorers compared (5-fold cross-validated, design-weighted)</h3>" + _tbl(sc_tab) +
                 "<h3>Deployed operating point</h3>" + _kv(val["operating_point"]) +
                 "<h3>Raw Laya at the configured pre-validation threshold</h3>" + _kv(val["laya_raw_at_config_threshold"]) +
                 "<h3>Strata</h3>" + _tbl(pd.DataFrame(val["strata"])))
        if val.get("categorical"):
            H.append("<h3>Categorical answers (storage setting / method / scale)</h3>" +
                     _tbl(pd.DataFrame(val["categorical"]).T.reset_index().rename(columns={"index": "question"})))
        if val.get("topic_flags"):
            H.append("<h3>Topic flags</h3>" + _tbl(pd.DataFrame(val["topic_flags"]).T.reset_index().rename(columns={"index": "flag"})))
        else:
            H.append("<p class='small'>Topic flags were not validated (no/too few human_&lt;topic&gt; labels); treat them as exploratory.</p>")
        if val.get("inter_rater"):
            H.append("<h3>Inter-rater agreement</h3>" + _kv(val["inter_rater"]))
        for n in ("val01_threshold_sweep", "val02_confusion", "val03_calibration", "val04_score_by_label"):
            i = _img(paths, n, paths.validation)
            if i:
                H.append(f"<figure>{i}</figure>")
    else:
        H.append("<p><i>No validation yet.</i></p>")
    cmp_f = paths.validation / "compare_summary.csv"
    if cmp_f.exists():
        H.append("<h3>Laya variants compared on the gold labels</h3>" + _tbl(pd.read_csv(cmp_f)))
    if seed:
        H.append("<h3>Collection recall on landmark (seed) papers</h3>" + _kv(seed))
    H.append("<h3>Data-quality checks</h3>" + _tbl(T["data_quality"]) + "<h3>Threshold sensitivity</h3>" + _tbl(T["threshold_sensitivity"]))
    H.append(_fig(paths, "fig22_laya_performance", "Laya compute performance.") + "<h3>Search-query yield</h3>" +
             _fig(paths, "fig23_query_yield", "Yield per query.") + _tbl(T["query_yield"]))

    H.append("<h2>9. Limitations</h2><ul>"
             "<li>Title/abstract only; papers screened on the title alone (no abstract recoverable) are flagged and routed to the review queue.</li>"
             "<li>Laya answers are probabilistic model outputs; the base checkpoints ship over-confident and are near chance on some zero-shot tasks "
             "(model card) - hence the calibration and human validation above. Categorical answers and topic flags are less validated than relevance.</li>"
             "<li>The English checkpoint reads ~320 tokens of text; longer abstracts are trimmed (flagged <code>trimmed</code>).</li>"
             "<li>Lexicon tags mean 'mentioned in abstract', not 'main topic'; absence of a tag is not evidence of absence.</li>"
             "<li>OpenAlex coverage, abstract availability and country metadata are incomplete; the latest months are under-indexed; "
             "the last year is partial (rates per month are compared, not totals).</li>"
             "<li>Single database (OpenAlex) plus abstract enrichment; no Scopus/Web of Science search was performed.</li>"
             "<li>Citation counts are strongly age-biased for 2025–2026 papers.</li>"
             "<li>Full texts were not analysed; no paywalls were bypassed.</li></ul>")
    H.append("<h2>10. Appendix: search queries</h2>" + _tbl(pd.DataFrame(cfg["queries"]).rename(columns={"q": "query"})))
    H.append("</body></html>")
    out = paths.report / "UHS_Literature_Report.html"
    out.write_text("\n".join(H), encoding="utf-8")
    log.info("Report: %s", out)
    return out
