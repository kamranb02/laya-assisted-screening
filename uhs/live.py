"""Live dashboard - run in a SECOND terminal while `collect` / `screen` run in the first.

    python run.py live            ->  http://127.0.0.1:8765   (auto-opens, auto-refreshes)

It opens the SQLite database READ-ONLY (WAL mode lets readers and the writer work at the same time),
so it can never disturb a running collection or Laya screening. Every `interval_seconds` it recomputes the
tables and a lightweight set of figures from whatever is in the database at that moment.
"""
from __future__ import annotations

import copy
import functools
import html
import http.server
import socketserver
import sqlite3
import threading
import time
import webbrowser
from types import SimpleNamespace

import pandas as pd

from . import analyze
from .figures import Figs

LIVE_FIGS = ["prisma", "yearly", "relevance", "threshold", "storage", "method", "scale", "topics",
              "coverage", "journals", "countries", "qy", "perf"]

CSS = """body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f4f6f9;color:#1b1f24}
header{background:#1F4E79;color:#fff;padding:.7rem 1.2rem;display:flex;justify-content:space-between;align-items:center}
main{padding:1rem 1.2rem;max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}
.box{background:#fff;border-radius:6px;padding:.8rem 1rem;box-shadow:0 1px 3px #0002}.box h3{margin:.1rem 0 .5rem;color:#1F4E79;font-size:1rem}
.kpi{display:flex;flex-wrap:wrap;gap:.6rem}.kpi div{background:#eef4fb;border-left:4px solid #0072B2;padding:.4rem .8rem;min-width:110px}
.kpi b{display:block;font-size:1.4rem}.bar{background:#dde5ee;border-radius:4px;height:14px;margin:2px 0 7px}.bar i{display:block;height:14px;background:#0072B2;border-radius:4px}
.bar.done i{background:#009E73}table{border-collapse:collapse;font-size:.8rem;width:100%}th{background:#1F4E79;color:#fff;text-align:left}td,th{border:1px solid #d5dbe3;padding:3px 6px}
img{max-width:100%}pre{background:#111;color:#cfe;padding:.5rem;font-size:.72rem;overflow:auto;max-height:230px;margin:0}.warn{color:#b35c00}
.small{font-size:.8rem;color:#555}"""


def _ro(paths):
    return sqlite3.connect(f"file:{paths.db}?mode=ro", uri=True, timeout=30)


def _tbl(df, n=15):
    if df is None or len(df) == 0:
        return "<p class='small'>no data yet</p>"
    return df.head(n).to_html(index=False, border=0, escape=True, float_format=lambda v: f"{v:.3g}", na_rep="")


def _progress(con, cfg, paths) -> str:
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    total = q("SELECT COUNT(*) FROM papers")
    dup = q("SELECT COUNT(*) FROM papers WHERE dup_of IS NOT NULL")
    elig = q("SELECT COUNT(*) FROM papers WHERE dup_of IS NULL AND eligibility='eligible'")
    s1 = q("SELECT COUNT(*) FROM laya WHERE stage>=1")
    s2 = q("SELECT COUNT(*) FROM laya WHERE stage=2")
    fail = q("SELECT COUNT(*) FROM laya WHERE stage=-1")
    errs = q("SELECT COUNT(*) FROM errors")
    kpi = [(f"{total:,}", "records collected"), (f"{dup:,}", "duplicates"), (f"{elig:,}", "eligible for Laya"),
           (f"{s1:,}", "Laya stage-1 done"), (f"{s2:,}", "stage-2 done"), (f"{fail}", "Laya failures"), (f"{errs}", "errors logged")]
    h = ["<div class='box'><h3>Pipeline progress</h3><div class='kpi'>" +
         "".join(f"<div><b>{a}</b>{b}</div>" for a, b in kpi) + "</div>"]
    # Laya ETA from measured throughput
    pf = paths.logs / "laya_performance.csv"
    if pf.exists():
        try:
            p = pd.read_csv(pf).tail(30)
            for st, g in p.groupby("stage"):
                rate = g.n_papers.sum() / max(g.seconds.sum(), 1e-9)
                left = (elig - s1) if st == 1 else max(0, q("SELECT COUNT(*) FROM laya WHERE stage=1 AND uhs_prob>=%s" % cfg["screening"]["stage2_min_prob"]) - s2)
                h.append(f"<p class='small'>Laya stage {st}: {rate:.2f} papers/s · remaining ≈ {left:,} · ETA ≈ {left / rate / 3600:.1f} h</p>")
        except Exception:
            pass
    h.append("</div><div class='box'><h3>Collection by query</h3>")
    for qid, n, api, done in con.execute("SELECT query_id,retrieved,api_count,done FROM query_state ORDER BY query_id"):
        pct = 100 if done else (min(100, 100 * n / api) if api else 0)
        h.append(f"<div class='small'>{qid}: {n:,} / {api if api is not None else '?'} {'✓' if done else ''}</div>"
                 f"<div class='bar {'done' if done else ''}'><i style='width:{pct:.0f}%'></i></div>")
    h.append("</div>")
    last = con.execute("SELECT ts,stage,ref,message FROM errors ORDER BY rowid DESC LIMIT 5").fetchall()
    if last:
        h.append("<div class='box'><h3 class='warn'>Latest errors</h3><table><tr><th>time</th><th>stage</th><th>ref</th><th>message</th></tr>" +
                 "".join("<tr>" + "".join(f"<td>{html.escape(str(x))[:90]}</td>" for x in r) + "</tr>" for r in last) + "</table></div>")
    lf = paths.logs / "pipeline.log"
    if lf.exists():
        tail = "".join(lf.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-14:])
        h.append(f"<div class='box'><h3>pipeline.log (tail)</h3><pre>{html.escape(tail)}</pre></div>")
    return "".join(h)


def snapshot(cfg, paths, log) -> str:
    """Recompute everything from the database; returns the path of the dashboard html."""
    live = paths.out / "live"
    (live / "fig").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H:%M:%S")
    body, notes = [], []
    if not paths.db.exists():
        body.append("<div class='box'>Waiting for the database… start `python run.py collect` in the other window.</div>")
    else:
        con = _ro(paths)
        try:
            body.append("<div class='grid'>" + _progress(con, cfg, paths) + "</div>")
            try:
                df = analyze.build_frame(con, cfg, paths)
                scr = int(df.decision.isin(["included", "excluded"]).sum())
                if scr == 0:
                    notes.append("No Laya-screened papers yet - analysis panels appear after screening starts "
                                 "(`python run.py screen`).")
                else:
                    P = analyze.prisma_counts(con, df, cfg)
                    T = analyze.make_tables(con, df, cfg)
                    lcfg = copy.deepcopy(cfg)
                    lcfg["analysis"].update(formats=["png"], dpi=110)
                    pf = paths.logs / "laya_performance.csv"
                    perf = pd.read_csv(pf) if pf.exists() else None
                    fg = Figs(lcfg, SimpleNamespace(figures=live / "fig"), log)
                    made = fg.all(df, T, P, perf, only=LIVE_FIGS)
                    body.append("<div class='grid'><div class='box'><h3>Key numbers (current threshold "
                                f"{P['threshold']} on {P['scorer']}, {'validated' if P['validated'] else 'not validated'})</h3><div class='kpi'>" +
                                "".join(f"<div><b>{v:,}</b>{k}</div>" for k, v in
                                        [("included", P["included"]), ("excluded by Laya", P["excluded_by_laya"]),
                                         ("auto-excluded", P["auto_excluded"]), ("pending", P["pending_screening"]),
                                         ("review queue", P["review_queue_size"])]) + "</div></div>"
                                f"<div class='box'><h3>Papers per year</h3>{_tbl(T['yearly'])}</div>"
                                f"<div class='box'><h3>Storage setting</h3>{_tbl(T['storage_type_counts'])}</div>"
                                f"<div class='box'><h3>Method</h3>{_tbl(T['method_counts'])}</div>"
                                f"<div class='box'><h3>Top journals</h3>{_tbl(T['top_journals'], 10)}</div>"
                                f"<div class='box'><h3>Top countries</h3>{_tbl(T['top_countries'], 10)}</div></div>")
                    body.append("<h2>Figures</h2><div class='grid'>" + "".join(
                        f"<div class='box'><img src='fig/{n}.png?t={int(time.time())}'></div>" for n in made) + "</div>")
            except Exception as e:  # partial data can break an analysis step - never kill the dashboard
                log.warning("live snapshot: %s", e)
                notes.append(f"Analysis step skipped this round: {html.escape(str(e))}")
        finally:
            con.close()
    page = (f"<html><head><meta charset='utf-8'><title>UHS live</title><style>{CSS}</style></head><body>"
            f"<header><b>UHS literature mining - LIVE</b><span>analysis updated {stamp} · page auto-refreshes</span></header><main>"
            + "".join(f"<p class='warn'>{n}</p>" for n in notes) + "".join(body) +
            "</main><script>setTimeout(()=>location.reload(),15000)</script></body></html>")
    out = live / "index.html"
    out.write_text(page, encoding="utf-8")
    return str(out)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def run_live(cfg, paths, log, port=None, interval=None, open_browser=True, once=False):
    port = int(port or cfg.get("live", {}).get("port", 8765))
    interval = int(interval or cfg.get("live", {}).get("interval_seconds", 60))
    live = paths.out / "live"
    live.mkdir(parents=True, exist_ok=True)
    snapshot(cfg, paths, log)
    if once:
        return str(live / "index.html")
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = None
    for p in range(port, port + 10):
        try:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", p), functools.partial(_Quiet, directory=str(live)))
            port = p
            break
        except OSError:
            continue
    if srv is None:
        raise RuntimeError("no free port for the live dashboard")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    log.info("LIVE dashboard at %s  (recomputed every %ds, Ctrl+C to stop)", url, interval)
    if open_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(interval)
            t = time.time()
            snapshot(cfg, paths, log)
            log.info("live snapshot refreshed in %.1fs", time.time() - t)
    except KeyboardInterrupt:
        log.info("live dashboard stopped")
    finally:
        srv.shutdown()
