"""Stage 3 - Laya screening & classification (two-stage, resumable, version-safe, performance-tracked).

Stage 1: the single `uhs` question for every eligible paper (one transformer row per paper).
Stage 2: the detail questions only for papers whose raw P(UHS) >= screening.stage2_min_prob.
Questions are independent (Laya answers each in its own row), so skipping stage 2 for clear negatives does not
change any other answer.

Safety rails
  * Every result stores a `model_tag` = laya version | checkpoint | question style | question hash | max_len.
    If the tag of the current configuration differs from what is already in the database, `screen` refuses to
    mix them (use `--rescreen`: the old table is backed up inside the database first).
  * A failing batch falls back to single-record mode; a failing record is marked (stage -1) and logged.
  * Title+abstract are composed with a token budget (`state_mode`), and trimmed records are flagged.
"""
from __future__ import annotations

import csv
import json
import time

from .common import get_meta, log_error, now, set_meta
from .laya_compat import AgentPool, compose_state
from .questions import Q_DETAIL, Q_UHS, QUESTIONS_ALL  # noqa: F401  (re-exported for other modules)

LAYA_COLS = ("openalex_id,stage,uhs_prob,answers_json,tokens,tokens_per_row,near_max_len,model_tag,screened_at,"
             "est_tokens,trimmed")


def _predict_safe(agent, states, qdefs, bs, con, ids, log):
    """Batch predict; on failure retry one-by-one so a single bad record cannot kill a multi-hour run."""
    try:
        return agent.predict(states, qdefs, bs)
    except Exception as e:
        log.warning("Batch failed (%s) - falling back to single-record mode", str(e)[:150])
    out = []
    for s, i in zip(states, ids):
        try:
            out.append(agent.predict([s], qdefs, 1)[0])
        except Exception as e2:
            log_error(con, "laya", i, e2)
            out.append(None)
    return out


def _perf_writer(paths):
    f = paths.logs / "laya_performance.csv"
    new = not f.exists()
    fh = open(f, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["timestamp", "stage", "n_papers", "seconds", "papers_per_sec", "tokens_per_row",
                    "batch_size", "max_len", "device", "route"])
    return fh, w


def _run_stage(con, pool, cfg, paths, log, stage, qdefs, rows, tag, deadline=None):
    from tqdm import tqdm
    sc = cfg["screening"]
    bs, maxc = int(sc["batch_size"]), int(sc["max_abstract_chars"])
    mode = sc.get("state_mode", "smart_trim")
    fh, w = _perf_writer(paths)
    done = 0
    try:
        for k in tqdm(range(0, len(rows), bs), desc=f"Laya stage {stage}", unit="batch"):
            if deadline and time.time() > deadline:
                log.info("Time budget reached - stopping cleanly (resume with the same command).")
                break
            chunk = rows[k:k + bs]
            by_route = {}
            for r in chunk:
                by_route.setdefault(pool.route(r[3]), []).append(r)
            for route, part in by_route.items():
                agent = pool.get(route)
                ids = [r[0] for r in part]
                composed = [compose_state(r[1], r[2], mode, agent.text_budget, maxc) for r in part]
                states = [c[0] for c in composed]
                t0 = time.perf_counter()
                results = _predict_safe(agent, states, qdefs, bs, con, ids, log)
                dt = time.perf_counter() - t0
                nq = len(qdefs)
                toks = []
                for r, res, (st, trimmed, est) in zip(part, results, composed):
                    if res is None:
                        if stage == 1:
                            con.execute(f"INSERT OR REPLACE INTO laya({LAYA_COLS}) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                        (r[0], -1, None, "{}", 0, 0, 0, tag, now(), est, trimmed))
                        continue
                    answers = res["answers"]
                    t_in = int(res.get("tokens", 0))
                    tpr = t_in / max(nq, 1)
                    toks.append(tpr)
                    if stage == 1:
                        p = float(answers["uhs"]["noul"])
                        con.execute(f"INSERT OR REPLACE INTO laya({LAYA_COLS}) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                    (r[0], 1, p, json.dumps(answers), t_in, tpr, int(tpr >= agent.max_len - 1),
                                     tag, now(), est, trimmed))
                    else:
                        old = con.execute("SELECT answers_json FROM laya WHERE openalex_id=?", (r[0],)).fetchone()
                        merged = json.loads(old[0] or "{}") if old else {}
                        merged.update(answers)
                        con.execute("UPDATE laya SET stage=2, answers_json=? WHERE openalex_id=?",
                                    (json.dumps(merged), r[0]))
                con.commit()
                done += len(part)
                w.writerow([now(), stage, len(part), round(dt, 3), round(len(part) / dt, 3) if dt else 0,
                            round(sum(toks) / len(toks), 1) if toks else 0, bs, agent.max_len, agent.device, route])
                fh.flush()
    finally:
        fh.close()
    return done


_S1 = """SELECT p.openalex_id,p.title,p.abstract,p.language FROM papers p
        LEFT JOIN laya l ON l.openalex_id=p.openalex_id
        WHERE l.openalex_id IS NULL AND p.dup_of IS NULL AND p.eligibility='eligible'
          AND (COALESCE(p.title,'')!='' OR COALESCE(p.abstract,'')!='')
        ORDER BY length(COALESCE(p.title,'')||COALESCE(p.abstract,''))"""
_S2 = """SELECT p.openalex_id,p.title,p.abstract,p.language FROM papers p JOIN laya l USING(openalex_id)
        WHERE l.stage=1 AND l.uhs_prob>=? AND p.dup_of IS NULL AND p.eligibility='eligible'
        ORDER BY length(COALESCE(p.title,'')||COALESCE(p.abstract,''))"""


def _check_tag(con, tag, rescreen, log):
    have = [r[0] for r in con.execute("SELECT DISTINCT model_tag FROM laya WHERE model_tag IS NOT NULL")]
    other = [t for t in have if t != tag]
    if not other:
        return
    if not rescreen:
        raise RuntimeError(
            "The database already holds Laya results produced with a different configuration:\n"
            f"   in database : {other}\n   current     : {tag}\n"
            "Mixing them would make the corpus inconsistent. Revert the config change, or re-run with "
            "`python run.py screen --rescreen` (the old results are kept in a backup table).")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    k, name = 1, f"laya_backup_{stamp}"
    while name in existing:
        k += 1
        name = f"laya_backup_{stamp}_{k}"
    stamp = name[len("laya_backup_"):]
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM laya")
    con.execute("DELETE FROM laya")
    con.commit()
    log.warning("Old Laya results moved to table laya_backup_%s; screening everything again.", stamp)


def screen(con, cfg, paths, log, pool=None, limit=None, max_hours=None, retry_failed=False, rescreen=False):
    sc = cfg["screening"]
    if not sc.get("enabled", True):
        log.info("screening disabled in config")
        return
    pool = pool or AgentPool(sc, log)
    tag = pool.tag()
    _check_tag(con, tag, rescreen, log)
    if retry_failed:
        con.execute("DELETE FROM laya WHERE stage=-1")
        con.commit()
    deadline = time.time() + max_hours * 3600 if max_hours else None
    s1 = con.execute(_S1).fetchall()
    s2 = con.execute(_S2, (sc["stage2_min_prob"],)).fetchall()
    log.info("Model tag: %s", tag)
    log.info("Pending: stage-1 = %d papers, stage-2 = %d papers so far (re-evaluated after stage 1; only raw P(UHS) >= %.2f)",
             len(s1), len(s2), sc["stage2_min_prob"])
    if limit:
        s1 = s1[:limit]
    if not s1 and not s2:
        log.info("Nothing to screen.")
        return
    set_meta(con, "screen_tag", tag)
    if not get_meta(con, "screen_first_started"):
        set_meta(con, "screen_first_started", now())
    if s1:
        _run_stage(con, pool, cfg, paths, log, 1, Q_UHS, s1, tag, deadline)
        s2 = con.execute(_S2, (sc["stage2_min_prob"],)).fetchall()
        if limit:
            s2 = s2[:limit]
    if s2 and not (deadline and time.time() > deadline):
        _run_stage(con, pool, cfg, paths, log, 2, Q_DETAIL, s2, tag, deadline)
    set_meta(con, "screen_provenance", pool.provenance())   # after loading, so loaded-agent info is included
    set_meta(con, "screen_last_finished", now())
    left = con.execute("""SELECT COUNT(*) FROM papers p LEFT JOIN laya l USING(openalex_id) WHERE l.openalex_id IS NULL
                          AND p.dup_of IS NULL AND p.eligibility='eligible'""").fetchone()[0]
    log.info("Screening finished. Eligible papers still without a Laya result: %d", left)
