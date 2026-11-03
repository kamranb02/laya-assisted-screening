"""Shared helpers: config, paths, logging, database schema, text utils."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import html
import json
import logging
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent
SECRET_KEYS = {"api_key"}


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _strip_secrets(x):
    if isinstance(x, dict):
        return {k: ("<hidden>" if k in SECRET_KEYS and v else _strip_secrets(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_secrets(v) for v in x]
    return x


def resolve_window(cfg: dict) -> tuple[str, str]:
    p = cfg["project"]
    start = str(p.get("start_date") or "2025-01-01")
    end = str(p.get("end_date") or "auto")
    if end.lower() == "auto":
        end = dt.date.today().isoformat()
    return start, end


def load_config(path=None) -> dict:
    """Load config.yaml, overlay an optional untracked config.local.yaml, resolve `end_date: auto`."""
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = p.with_name("config.local.yaml")
    if local.exists():
        with open(local, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    cfg["_path"] = str(p)
    start, end = resolve_window(cfg)
    cfg["_window"] = (start, end)
    cfg["_start_year"], cfg["_end_year"] = int(start[:4]), int(end[:4])
    # the hash covers every setting except secrets and the volatile 'auto' end date
    stable = _strip_secrets({k: v for k, v in cfg.items() if not k.startswith("_")})
    stable["project"]["end_date"] = str(cfg["project"].get("end_date"))
    cfg["_hash"] = hashlib.sha1(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return cfg


def make_paths(cfg: dict) -> SimpleNamespace:
    out = ROOT / cfg["paths"]["output_dir"]
    data = ROOT / cfg["paths"]["data_dir"]
    p = SimpleNamespace(
        root=ROOT, data=data, db=data / "uhs_literature.db", out=out,
        tables=out / "tables", figures=out / "figures", report=out / "report",
        validation=out / "validation", logs=out / "logs", excel=out / "excel", provenance=out / "provenance",
    )
    for d in (p.data, p.tables, p.figures, p.report, p.validation, p.logs, p.excel, p.provenance):
        d.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(paths) -> logging.Logger:
    log = logging.getLogger("uhs")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(paths.logs / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
  openalex_id TEXT PRIMARY KEY, doi TEXT, title TEXT, abstract TEXT, abstract_source TEXT,
  authors TEXT, publication_year INTEGER, publication_date TEXT, type TEXT, language TEXT,
  journal TEXT, publisher TEXT, cited_by_count INTEGER, is_oa INTEGER, oa_status TEXT,
  is_retracted INTEGER DEFAULT 0, landing_page_url TEXT, pdf_url TEXT,
  countries TEXT, institutions TEXT, keywords TEXT, primary_topic TEXT,
  first_query TEXT, source TEXT, first_seen_at TEXT,
  dup_of TEXT, eligibility TEXT DEFAULT 'eligible', exclude_reason TEXT,
  crossref_checked INTEGER DEFAULT 0, s2_checked INTEGER DEFAULT 0, meta_backfilled INTEGER DEFAULT 0,
  tags TEXT, tools TEXT, rule_uhs TEXT, rule_storage TEXT, rule_scale TEXT, rule_method TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(publication_year);
CREATE TABLE IF NOT EXISTS paper_queries(
  openalex_id TEXT, query_id TEXT, PRIMARY KEY(openalex_id, query_id));
CREATE TABLE IF NOT EXISTS query_state(
  query_id TEXT PRIMARY KEY, query TEXT, cursor TEXT, retrieved INTEGER DEFAULT 0,
  api_count INTEGER, done INTEGER DEFAULT 0, updated_at TEXT);
CREATE TABLE IF NOT EXISTS search_log(
  run_at TEXT, query_id TEXT, query TEXT, api_count INTEGER, retrieved INTEGER,
  new_unique INTEGER, status TEXT, seconds REAL);
CREATE TABLE IF NOT EXISTS laya(
  openalex_id TEXT PRIMARY KEY, stage INTEGER, uhs_prob REAL, answers_json TEXT,
  tokens INTEGER, tokens_per_row REAL, near_max_len INTEGER, model_tag TEXT, screened_at TEXT,
  est_tokens INTEGER DEFAULT 0, trimmed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS manual(
  openalex_id TEXT PRIMARY KEY, decision INTEGER, source TEXT, note TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS errors(ts TEXT, stage TEXT, ref TEXT, message TEXT);
"""


def connect(paths) -> sqlite3.Connection:
    con = sqlite3.connect(str(paths.db))
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    con.commit()
    return con


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_error(con, stage, ref, msg):
    con.execute("INSERT INTO errors VALUES(?,?,?,?)", (now(), stage, str(ref), str(msg)[:1000]))
    con.commit()


def set_meta(con, key, value):
    con.execute("INSERT OR REPLACE INTO meta VALUES(?,?,?)",
                (key, json.dumps(value, default=str) if not isinstance(value, str) else value, now()))
    con.commit()


def get_meta(con, key, default=None):
    r = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if not r:
        return default
    try:
        return json.loads(r[0])
    except Exception:
        return r[0]


# ----------------------------------------------------------------- text utils
def norm_doi(x) -> str:
    x = str(x or "").lower().strip()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", x).strip(" .;")


def norm_title(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).lower()
    return re.sub(r"[^a-z0-9]", "", t)


def short_id(oa_id: str) -> str:
    return str(oa_id or "").rsplit("/", 1)[-1]


def abstract_from_inverted(inv) -> str:
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(s) -> str:
    if s is None:
        return ""
    s = html.unescape(str(s))
    s = _ILLEGAL.sub("", s)
    return re.sub(r"\s+", " ", s).strip()
