"""Stage 5b - CSV + formatted Excel workbook exports."""
from __future__ import annotations

import json

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .common import clean_text
from .lexicon import TAGS

MAIN_COLS = ["openalex_id", "doi", "publication_year", "pub_month", "title", "authors", "journal", "publisher", "type",
             "is_preprint", "language", "cited_by_count", "is_oa", "oa_status", "countries", "institutions", "keywords",
             "primary_topic", "decision", "decision_source", "score_final", "uhs_prob", "confidence_tier", "review_priority",
             "laya_storage_type", "laya_storage_type_conf", "laya_method", "laya_method_conf", "laya_scale",
             "laya_scale_conf", "laya_multiphase_flow", "laya_geochemistry", "laya_biology_methanogenesis",
             "laya_geomechanics_integrity", "laya_experimental", "laya_numerical", "laya_data_driven_ml",
             "rule_uhs", "rule_storage", "rule_method", "rule_scale", "tags", "tools", "abstract_missing",
             "abstract_source", "trimmed", "landing_page_url", "pdf_url", "abstract"]
CLASS_COLS = [c for c in MAIN_COLS if c not in ("abstract", "authors", "institutions", "landing_page_url", "pdf_url", "publisher")]


def _clean_df(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda v: clean_text(v)[:32000] if isinstance(v, str) else v)
    return df


def _style(ws, widths=None):
    hf = PatternFill("solid", fgColor="1F4E79")
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF", name="Arial"), hf
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for i, col in enumerate(ws.columns, 1):
        head = str(col[0].value or "")
        w = (widths or {}).get(head)
        if w is None:
            mx = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
            w = min(60, max(9, mx + 2))
            if head in ("abstract",):
                w = 80
        ws.column_dimensions[get_column_letter(i)].width = w


def write_outputs(df, T, P, cfg, paths, log):
    inc = df[df.decision == "included"]
    exc = df[df.decision != "included"]
    cols = [c for c in MAIN_COLS if c in df.columns]
    ord_ = ["publication_year", "cited_by_count"]
    csvs = {
        "papers_all.csv": df[cols + ["exclude_reason", "dup_of"]],
        "included.csv": inc.sort_values(ord_, ascending=False)[cols],
        "excluded.csv": exc[cols + ["exclude_reason", "dup_of"]],
        "classification.csv": inc[[c for c in CLASS_COLS if c in df.columns]],
        "tags_matrix.csv": inc[["openalex_id", "doi", "publication_year", "title"] + [f"tag_{t}" for t in TAGS]].astype(
            {f"tag_{t}": int for t in TAGS}),
    }
    for k, v in csvs.items():
        v.to_csv(paths.tables / k, index=False, encoding="utf-8-sig")
    for k, v in T.items():
        v.to_csv(paths.tables / f"{k}.csv", index=False, encoding="utf-8-sig")
    (paths.tables / "prisma_counts.json").write_text(json.dumps(P, indent=2), encoding="utf-8")

    x = paths.excel / f"UHS_Literature_{cfg['_window'][0][:4]}_{cfg['_window'][1][:4]}.xlsx"
    readme = pd.DataFrame({"UHS literature mining - workbook guide": [
        f"Project: {cfg['project']['name']}",
        f"Include rule: scorer={P['scorer']} >= {P['threshold']} ({'VALIDATED against human gold labels' if P['validated'] else 'NOT VALIDATED - raw rule'}); borderline band {P['review_band']}",
        "Included / Excluded: full records with Laya + rule-based classification.",
        "Review_Queue: papers where Laya and keyword rules disagree, the score is borderline, or only the title was available. Fill human_decision (1/0) and run `python run.py ingest-review`.",
        "Tags_Matrix: 0/1 lexicon tags (abstract mentions) for pivoting in Excel.",
        "review_priority: A = possible false negative, B = possible false positive, C = rules disagree, D = borderline.",
        "Scores are model outputs, not ground truth - see the Validation section of the HTML report. decision_source=human marks papers decided by a person (gold sample / review queue).",
        "Abstract-only analysis: 'no mention' does not mean 'not studied'."]})
    sheets = {
        "README": readme, "PRISMA": pd.DataFrame([(k, str(v)) for k, v in P.items()], columns=["item", "value"]),
        "Included": csvs["included.csv"], "Excluded": csvs["excluded.csv"], "Review_Queue": T["review_queue"],
        "Classification": csvs["classification.csv"], "Tags_Matrix": csvs["tags_matrix.csv"],
        "Yearly": T["yearly"], "Monthly": T["monthly"], "Topic_Shift": T.get("topic_shift", pd.DataFrame()), "Storage_x_Method": T["storage_x_method"], "Storage_x_Scale": T["storage_x_scale"],
        "Coverage_Matrix": T["coverage_matrix"], "Coverage_Gaps": T["coverage_gaps"],
        "Threshold_Sensitivity": T["threshold_sensitivity"], "Top_Journals": T["top_journals"],
        "Top_Countries": T["top_countries"], "Top_Cited": T["top_cited"], "Query_Yield": T["query_yield"],
        "Data_Quality": T["data_quality"]}
    with pd.ExcelWriter(x, engine="openpyxl") as w:
        for name, d in sheets.items():
            _clean_df(d).to_excel(w, sheet_name=name[:31], index=False)
        for name in sheets:
            ws = w.sheets[name[:31]]
            _style(ws, {"title": 60, "openalex_id": 26, "doi": 30} if name in ("Included", "Excluded", "Review_Queue") else None)
            if name == "README":
                ws.column_dimensions["A"].width = 120
    log.info("Excel workbook: %s", x)
    return x
