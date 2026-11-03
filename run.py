"""UHS literature-mining pipeline v3 - command line.

Typical order (see GUIDE_FROM_SCRATCH.md):
    python run.py doctor               # environment, OpenAlex key + budget, Laya version
    python run.py smoke                # Laya sanity + style recommendation + context-budget check
    python run.py estimate             # how many records will each query fetch? (1 request per query)
    python run.py collect              # OpenAlex collection (resumable, budget-aware)
    python run.py process              # dedup + abstract recovery (Crossref, Semantic Scholar) + rules + tags
    python run.py sample               # BLIND gold labeling sheet (before screening; independent of Laya)
    python run.py sample --second-rater   # optional blind copy of a share of the sheet for a 2nd reviewer
    python run.py compare              # rank Laya checkpoints / question styles on YOUR gold labels
    python run.py screen               # Laya two-stage screening (resumable; refuses to mix configurations)
    python run.py evaluate             # metrics + calibration + operating point (validated threshold)
    python run.py ingest-review FILE   # human decisions on the review queue -> database
    python run.py analyze              # tables, figures, Excel, HTML report
    python run.py recall               # collection/pipeline recall on landmark (seed) papers
    python run.py export-finetune      # gold labels -> JSONL for fine-tuning a Laya checkpoint
    python run.py freeze               # provenance bundle (config, versions, database hash)
    python run.py status | live | all
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")          # Laya model card: TensorFlow's abseil can deadlock model construction

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import pandas as pd  # noqa: E402

from uhs import analyze, collect, export, figures, laya_compat, live, process, report, screen, smoke, validate  # noqa: E402
from uhs import __version__  # noqa: E402
from uhs.common import connect, get_logger, get_meta, load_config, make_paths, set_meta, _strip_secrets  # noqa: E402


def do_process(con, cfg, log, enrich=True):
    process.dedup(con, log)
    if enrich:
        collect.enrich_crossref(con, cfg, log)
        collect.enrich_semantic_scholar(con, cfg, log)
        process.dedup(con, log)            # an enriched abstract can change which duplicate is the better record
    return process.apply_rules(con, cfg, log)


def do_freeze(con, cfg, paths, log):
    """Write everything needed to reproduce/describe the run (PRISMA-trAIce style provenance)."""
    h = hashlib.sha256()
    with open(paths.db, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    info = {"pipeline_version": __version__, "created": pd.Timestamp.now().isoformat(), "config_hash": cfg["_hash"],
            "window": cfg["_window"], "database_sha256": h.hexdigest(), "database_bytes": paths.db.stat().st_size,
            "laya_versions": laya_compat.installed_versions(), "screen_tag": get_meta(con, "screen_tag"),
            "screen_provenance": get_meta(con, "screen_provenance"),
            "counts": {k: con.execute(q).fetchone()[0] for k, q in {
                "papers": "SELECT COUNT(*) FROM papers", "laya_rows": "SELECT COUNT(*) FROM laya",
                "manual_decisions": "SELECT COUNT(*) FROM manual"}.items()}}
    (paths.provenance / "provenance.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    (paths.provenance / "config_used.json").write_text(
        json.dumps(_strip_secrets({k: v for k, v in cfg.items() if not k.startswith("_")}), indent=2, default=str), encoding="utf-8")
    for f in ("operating_point.json", "calibration.json", "metrics.json", "compare_summary.csv", "smoke_report.json"):
        if (paths.validation / f).exists():
            shutil.copy(paths.validation / f, paths.provenance / f)
    log.info("Provenance bundle written to %s", paths.provenance)


def do_analyze(con, cfg, paths, log, make_report=True):
    df = analyze.build_frame(con, cfg, paths, log)
    if df.empty:
        log.error("Database is empty - run `collect` first.")
        return
    P = analyze.prisma_counts(con, df, cfg)
    T = analyze.make_tables(con, df, cfg)
    perf_f = paths.logs / "laya_performance.csv"
    perf = pd.read_csv(perf_f) if perf_f.exists() else None
    figures.Figs(cfg, paths, log).all(df, T, P, perf)
    export.write_outputs(df, T, P, cfg, paths, log)
    validate.recall_check(con, cfg, paths, log)
    do_freeze(con, cfg, paths, log)
    if make_report:
        report.build_report(df, T, P, cfg, paths, log, con=con)
    log.info("PRISMA: %s", json.dumps({k: v for k, v in P.items() if not isinstance(v, (dict, list))}))
    if not P["validated"]:
        log.warning("The classification is NOT validated yet: run `sample`, label, then `evaluate`.")


def do_status(con, cfg, paths, log):
    def q(sql):
        return con.execute(sql).fetchone()[0]
    auto_sql = "SELECT COUNT(*) FROM papers WHERE dup_of IS NULL AND eligibility != 'eligible'"
    noabs_sql = "SELECT COUNT(*) FROM papers WHERE COALESCE(abstract, '') = ''"
    print(f"window                  : {cfg['_window'][0]} .. {cfg['_window'][1]}")
    print(f"papers total            : {q('SELECT COUNT(*) FROM papers'):,}")
    print(f"  duplicates flagged    : {q('SELECT COUNT(*) FROM papers WHERE dup_of IS NOT NULL'):,}")
    print(f"  auto-excluded (rules) : {q(auto_sql):,}")
    print(f"  missing abstract      : {q(noabs_sql):,}")
    print(f"laya stage>=1           : {q('SELECT COUNT(*) FROM laya WHERE stage>=1'):,}")
    print(f"laya stage 2 (detailed) : {q('SELECT COUNT(*) FROM laya WHERE stage=2'):,}")
    print(f"laya failed             : {q('SELECT COUNT(*) FROM laya WHERE stage=-1'):,}")
    print(f"manual decisions        : {q('SELECT COUNT(*) FROM manual'):,}")
    print(f"errors logged           : {q('SELECT COUNT(*) FROM errors'):,}")
    print(f"screen tag              : {get_meta(con, 'screen_tag')}")
    for r in con.execute("SELECT query_id,retrieved,api_count,done FROM query_state ORDER BY query_id"):
        print(f"  query {r[0]}: {r[1]:,}/{r[2]} {'done' if r[3] else 'INCOMPLETE'}")


def do_doctor(cfg, paths, log):
    import importlib
    import platform
    print(f"Python {platform.python_version()} on {platform.system()}")
    for m in ("numpy", "pandas", "matplotlib", "openpyxl", "tqdm", "yaml", "requests", "sklearn", "torch", "laya"):
        try:
            mod = importlib.import_module(m)
            print(f"  [ok]   {m} {getattr(mod, '__version__', '')}")
        except Exception as e:
            print(f"  [{'warn' if m == 'sklearn' else 'FAIL'}] {m}: {e}"
                  + ("  (needed for the TF-IDF baseline / stacked scorer)" if m == "sklearn" else ""))
    print("Laya runtime:", json.dumps(laya_compat.installed_versions()))
    print(f"Configured checkpoint: {cfg['screening']['checkpoint']} ({cfg['screening']['model_id']}), "
          f"question style: {cfg['screening']['question_style']}")
    cl = collect.Client(cfg, log)
    key = cl.api_key
    print(f"OpenAlex key ({cfg['openalex']['api_key_env']}): {'set (' + key[:4] + '***' + ')' if key else 'MISSING - create a free key at openalex.org/settings/api'}")
    try:
        d = cl.get_json("https://api.openalex.org/rate-limit", {"api_key": key}) if key else None
        print("  rate-limit/budget:", json.dumps((d or {}).get("rate_limit", d))[:300])
        p = collect.build_params(cfg["queries"][0], cfg, "*", cl)
        p["per-page"] = 1
        r = cl.get_json(cfg["openalex"]["base_url"], p)
        print(f"  [ok] test query {cfg['queries'][0]['id']} ({cfg['openalex']['search_filter']}) matches {r['meta']['count']:,} records "
              f"in {cfg['_window'][0]} .. {cfg['_window'][1]}")
    except Exception as e:
        print(f"  [FAIL] OpenAlex request: {e}\n  -> if the message says the filter is unknown, change openalex.search_filter in config.yaml")
    print(f"Database: {paths.db} ({'exists' if paths.db.exists() else 'will be created on first run'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["smoke", "collect", "process", "screen", "analyze", "report", "sample", "compare",
                                        "evaluate", "recall", "status", "all", "doctor", "estimate", "live", "ingest-review",
                                        "export-finetune", "freeze"])
    ap.add_argument("file", nargs="?", help="ingest-review: the filled review queue (csv/xlsx)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--refresh", action="store_true", help="collect: re-run all queries from scratch")
    ap.add_argument("--only", nargs="*", help="collect: only these query ids (e.g. q01 q05)")
    ap.add_argument("--no-enrich", action="store_true", help="process: skip Crossref / Semantic Scholar abstract recovery")
    ap.add_argument("--limit", type=int, help="screen: max papers (for testing)")
    ap.add_argument("--max-hours", type=float, help="screen: stop cleanly after N hours")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--rescreen", action="store_true", help="screen: discard (back up) old Laya results made with another configuration")
    ap.add_argument("--n", type=int, help="sample: number of papers")
    ap.add_argument("--design", choices=["rules", "laya"], help="sample: stratification design")
    ap.add_argument("--second-rater", action="store_true", help="sample: write a blind second-rater sheet")
    ap.add_argument("--force", action="store_true", help="sample: overwrite existing labeling sheet (DANGER)")
    ap.add_argument("--variants", nargs="*", help="compare: variant names from config.compare.variants")
    ap.add_argument("--port", type=int, help="live: dashboard port")
    ap.add_argument("--interval", type=int, help="live: seconds between recomputations")
    ap.add_argument("--no-browser", action="store_true", help="live: do not open the browser")
    ap.add_argument("--once", action="store_true", help="live: write one snapshot and exit")
    ap.add_argument("--quick", action="store_true", help="smoke: skip the throughput benchmark")
    a = ap.parse_args()

    cfg = load_config(a.config)
    paths = make_paths(cfg)
    log = get_logger(paths)
    con = connect(paths)
    c = a.command
    try:
        if c == "smoke":
            r = smoke.run_smoke(cfg, paths, log, con=con, quick=a.quick)
            sys.exit(0 if r["overall_pass"] else 2)
        elif c == "collect":
            collect.collect(con, cfg, log, refresh=a.refresh, only=a.only)
        elif c == "process":
            do_process(con, cfg, log, enrich=not a.no_enrich)
        elif c == "screen":
            do_process(con, cfg, log, enrich=False)
            screen.screen(con, cfg, paths, log, limit=a.limit, max_hours=a.max_hours, retry_failed=a.retry_failed,
                          rescreen=a.rescreen)
        elif c in ("analyze", "report"):
            do_analyze(con, cfg, paths, log)
        elif c == "sample":
            validate.make_sample(con, cfg, paths, log, n=a.n, force=a.force, design=a.design, second_rater=a.second_rater)
        elif c == "compare":
            validate.compare(con, cfg, paths, log, names=a.variants)
        elif c == "evaluate":
            validate.evaluate(con, cfg, paths, log)
            do_analyze(con, cfg, paths, log)
        elif c == "ingest-review":
            if not a.file:
                sys.exit("usage: python run.py ingest-review output/tables/review_queue.csv")
            validate.ingest_review(con, a.file, log)
            do_analyze(con, cfg, paths, log)
        elif c == "export-finetune":
            validate.export_finetune(con, cfg, paths, log)
        elif c == "recall":
            validate.recall_check(con, cfg, paths, log)
        elif c == "freeze":
            do_freeze(con, cfg, paths, log)
        elif c == "status":
            do_status(con, cfg, paths, log)
        elif c == "doctor":
            do_doctor(cfg, paths, log)
        elif c == "estimate":
            collect.estimate(cfg, log)
        elif c == "live":
            live.run_live(cfg, paths, log, port=a.port, interval=a.interval, open_browser=not a.no_browser, once=a.once)
        elif c == "all":
            if collect.collect(con, cfg, log):
                do_process(con, cfg, log)
                screen.screen(con, cfg, paths, log, max_hours=a.max_hours)
                do_analyze(con, cfg, paths, log)
            else:
                log.error("Collection incomplete (budget or errors) - `all` stops before screening. Re-run when resolved.")
    except (RuntimeError, FileNotFoundError, FileExistsError, ValueError) as e:
        log.error("%s", e)
        sys.exit(2)
    finally:
        con.close()


if __name__ == "__main__":
    main()
