"""Stage 1 - collection from OpenAlex + abstract enrichment (Crossref, optional Semantic Scholar).

Design notes
  * HTTP errors / rate limits are retried with back-off. A daily-budget error (OpenAlex meters usage since
    Feb 2026) stops the run cleanly; the cursor is saved, so the same command resumes after the reset.
  * No result cap unless max_results_per_query > 0; every (paper, query) hit is stored (provenance, query yield).
  * The publication window comes from project.start_date / end_date (end_date: auto = today).
  * Missing abstracts are recovered from Crossref and Semantic Scholar BEFORE screening, because a decision made
    on the title alone is unreliable (the record is flagged `abstract_missing` otherwise).
"""
from __future__ import annotations

import os
import re
import time

import requests

from .common import (abstract_from_inverted, clean_text, log_error, norm_doi, now)

SELECT = ("id,doi,title,publication_year,publication_date,type,language,primary_location,"
          "best_oa_location,open_access,authorships,cited_by_count,is_retracted,"
          "abstract_inverted_index,keywords,primary_topic")

COLS = ["openalex_id", "doi", "title", "abstract", "abstract_source", "authors", "publication_year",
        "publication_date", "type", "language", "journal", "publisher", "cited_by_count", "is_oa",
        "oa_status", "is_retracted", "landing_page_url", "pdf_url", "countries", "institutions",
        "keywords", "primary_topic", "first_query", "source", "first_seen_at", "meta_backfilled"]


class BudgetExhausted(RuntimeError):
    """The OpenAlex daily credit budget is used up (resume after the reset, midnight UTC)."""


_BUDGET_WORDS = re.compile(r"budget|credit|insufficient|quota|usage limit|payment", re.I)


class Client:
    """Tiny resilient JSON HTTP client (GET and POST)."""

    def __init__(self, cfg, log, api_key=None):
        self.cfg, self.log = cfg, log
        self.s = requests.Session()
        oa = cfg["openalex"]
        self.api_key = api_key if api_key is not None else (
            os.environ.get(oa.get("api_key_env", ""), "") or oa.get("api_key", ""))
        mail = oa.get("mailto") or ""
        self.s.headers["User-Agent"] = f"UHS-LitMine/3.0 (mailto:{mail})" if mail else "UHS-LitMine/3.0"
        self.last_credits = None

    def _request(self, method, url, params=None, json=None, headers=None):
        oa = self.cfg["openalex"]
        last = None
        for attempt in range(int(oa["max_retries"])):
            try:
                r = self.s.request(method, url, params=params, json=json, headers=headers, timeout=oa["timeout"])
                if "X-RateLimit-Credits-Used" in r.headers:
                    self.last_credits = r.headers.get("X-RateLimit-Credits-Used")
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return None
                body = r.text[:400]
                if r.status_code in (402, 403, 429) and "openalex.org" in url and _BUDGET_WORDS.search(body) \
                        and oa.get("stop_on_budget_exhausted", True):
                    raise BudgetExhausted(f"HTTP {r.status_code}: {body}")
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = float(r.headers.get("Retry-After", 0) or 0) or min(90, 2 ** (attempt + 1))
                    self.log.warning("HTTP %s from %s - retry %d in %.0fs", r.status_code, url, attempt + 1, wait)
                    last = f"HTTP {r.status_code}"
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {body}")
            except (requests.ConnectionError, requests.Timeout) as e:
                last = str(e)
                time.sleep(min(90, 2 ** (attempt + 1)))
        raise RuntimeError(f"giving up after {oa['max_retries']} attempts ({last})")

    def get_json(self, url, params=None):
        return self._request("GET", url, params=params)

    def post_json(self, url, json=None, params=None, headers=None):
        return self._request("POST", url, params=params, json=json, headers=headers)


# --------------------------------------------------------------------- parsing
def work_to_row(w: dict, query_id: str, source: str = "OpenAlex") -> dict:
    pl = w.get("primary_location") or {}
    best = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}
    src = pl.get("source") or {}
    auth = w.get("authorships") or []
    names = [((a.get("author") or {}).get("display_name") or "") for a in auth]
    countries = sorted({c for a in auth for c in (a.get("countries") or []) if c})
    insts = []
    for a in auth:
        for i in a.get("institutions") or []:
            n = i.get("display_name")
            if n and n not in insts:
                insts.append(n)
    kws = [k.get("display_name") for k in (w.get("keywords") or []) if k.get("display_name")]
    abstract = clean_text(abstract_from_inverted(w.get("abstract_inverted_index")))
    return dict(
        openalex_id=w.get("id", ""), doi=norm_doi(w.get("doi")), title=clean_text(w.get("title")),
        abstract=abstract, abstract_source="openalex" if abstract else "none",
        authors="; ".join(n for n in names if n), publication_year=w.get("publication_year"),
        publication_date=w.get("publication_date"), type=w.get("type"), language=w.get("language"),
        journal=src.get("display_name", "") or "", publisher=src.get("host_organization_name", "") or "",
        cited_by_count=w.get("cited_by_count", 0) or 0, is_oa=int(bool(oa.get("is_oa", False))),
        oa_status=oa.get("oa_status", "") or "", is_retracted=int(bool(w.get("is_retracted"))),
        landing_page_url=best.get("landing_page_url") or pl.get("landing_page_url") or "",
        pdf_url=best.get("pdf_url") or pl.get("pdf_url") or "",
        countries=";".join(countries), institutions="; ".join(insts[:30]), keywords="; ".join(kws),
        primary_topic=(w.get("primary_topic") or {}).get("display_name", "") or "",
        first_query=query_id, source=source, first_seen_at=now(), meta_backfilled=1,
    )


_UPSERT = f"""INSERT INTO papers({",".join(COLS)}) VALUES({",".join("?" * len(COLS))})
ON CONFLICT(openalex_id) DO UPDATE SET
  cited_by_count=excluded.cited_by_count, is_oa=excluded.is_oa, oa_status=excluded.oa_status,
  is_retracted=excluded.is_retracted, language=excluded.language,
  landing_page_url=COALESCE(NULLIF(excluded.landing_page_url,''), papers.landing_page_url),
  pdf_url=COALESCE(NULLIF(excluded.pdf_url,''), papers.pdf_url),
  countries=excluded.countries, institutions=excluded.institutions, keywords=excluded.keywords,
  primary_topic=excluded.primary_topic, meta_backfilled=1,
  abstract=CASE WHEN COALESCE(papers.abstract,'')='' THEN excluded.abstract ELSE papers.abstract END,
  abstract_source=CASE WHEN COALESCE(papers.abstract,'')='' THEN excluded.abstract_source ELSE papers.abstract_source END
"""


def upsert(con, rows, query_id=None) -> int:
    """Insert/update rows; returns number of NEW unique papers."""
    if not rows:
        return 0
    ids = [r["openalex_id"] for r in rows]
    have = set()
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        have |= {x[0] for x in con.execute(
            f"SELECT openalex_id FROM papers WHERE openalex_id IN ({','.join('?' * len(chunk))})", chunk)}
    new = len({i for i in ids if i not in have})
    con.executemany(_UPSERT, [tuple(r[c] for c in COLS) for r in rows])
    if query_id:
        con.executemany("INSERT OR IGNORE INTO paper_queries VALUES(?,?)", [(i, query_id) for i in ids])
    con.commit()
    return new


def build_params(q: dict, cfg: dict, cursor: str, client: Client) -> dict:
    y0, y1 = cfg["_window"]
    oa = cfg["openalex"]
    date = f"from_publication_date:{y0},to_publication_date:{y1}"
    text = re.sub(r"\s+", " ", q["q"].replace(",", " ")).strip()
    mode = q.get("mode", oa.get("search_mode", "title_abstract"))
    sf = oa.get("search_filter", "title_and_abstract.search")
    p = ({"filter": f"{date},{sf}:{text}"} if mode == "title_abstract" else {"filter": date, "search": text})
    p.update({"per-page": min(200, int(oa["per_page"])), "cursor": cursor, "select": SELECT})
    if client.api_key:
        p["api_key"] = client.api_key
    if oa.get("mailto"):
        p["mailto"] = oa["mailto"]
    return p


# ------------------------------------------------------------------- collection
def collect(con, cfg, log, client=None, refresh=False, only=None):
    client = client or Client(cfg, log)
    oa = cfg["openalex"]
    cap = int(oa.get("max_results_per_query") or 0)
    if not client.api_key:
        log.warning("No OpenAlex API key (env %s). Since Feb 2026 a free key is required for real use; "
                    "without one only a handful of calls work.", oa.get("api_key_env"))
    log.info("Publication window: %s .. %s", *cfg["_window"])
    for q in cfg["queries"]:
        qid = q["id"]
        if only and qid not in only:
            continue
        st = con.execute("SELECT cursor,retrieved,done,api_count FROM query_state WHERE query_id=?", (qid,)).fetchone()
        if st is None or refresh:
            cursor, n, api_count = "*", 0, None
            con.execute("INSERT OR REPLACE INTO query_state VALUES(?,?,?,?,?,?,?)",
                        (qid, q["q"], cursor, 0, None, 0, now()))
        elif st[2]:
            log.info("[%s] already complete (%d records) - skipped (use --refresh to redo)", qid, st[1])
            continue
        else:
            cursor, n, api_count = st[0], st[1], st[3]
            log.info("[%s] resuming at %d records", qid, n)
        t0, new_total, status, note = time.time(), 0, "ok", ""
        try:
            while cursor:
                d = client.get_json(oa["base_url"], build_params(q, cfg, cursor, client))
                if d is None:
                    raise RuntimeError("empty response")
                if api_count is None:
                    api_count = (d.get("meta") or {}).get("count")
                batch = d.get("results") or []
                rows = [work_to_row(w, qid) for w in batch]
                new_total += upsert(con, rows, qid)
                n += len(batch)
                cursor = (d.get("meta") or {}).get("next_cursor") if batch else None
                con.execute("UPDATE query_state SET cursor=?,retrieved=?,api_count=?,updated_at=? WHERE query_id=?",
                            (cursor, n, api_count, now(), qid))
                con.commit()
                log.info("[%s] %d / %s", qid, n, api_count)
                if cap and n >= cap:
                    status = "TRUNCATED_BY_CAP" if (api_count or 0) > n else "ok"
                    break
                time.sleep(float(oa["sleep_seconds"]))
            if status == "ok":
                con.execute("UPDATE query_state SET done=1 WHERE query_id=?", (qid,))
                if api_count is not None and n != api_count:
                    note = f" (retrieved {n} vs reported count {api_count})"
        except BudgetExhausted as e:
            status = "BUDGET_EXHAUSTED"
            log.error("[%s] OpenAlex daily budget exhausted - stopping. Re-run `collect` after the reset "
                      "(midnight UTC); the cursor is saved. (%s)", qid, str(e)[:200])
            con.execute("INSERT INTO search_log VALUES(?,?,?,?,?,?,?,?)",
                        (now(), qid, q["q"], api_count, n, new_total, status, round(time.time() - t0, 1)))
            con.commit()
            return False
        except Exception as e:  # keep going with the next query; cursor is saved for resume
            status = f"ERROR: {e}"
            log.error("[%s] %s", qid, e)
            log_error(con, "collect", qid, e)
        con.execute("INSERT INTO search_log VALUES(?,?,?,?,?,?,?,?)",
                    (now(), qid, q["q"], api_count, n, new_total, status + note, round(time.time() - t0, 1)))
        con.commit()
    bad = con.execute("SELECT query_id FROM query_state WHERE done=0").fetchall()
    if bad:
        log.warning("Incomplete queries (re-run `collect` to resume): %s", [b[0] for b in bad])
    return not bad


# ------------------------------------------------------------ abstract recovery
def _strip_jats(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = clean_text(s)
    return re.sub(r"^abstract[:.\s]+", "", s, flags=re.I)


def enrich_crossref(con, cfg, log, client=None, limit=None):
    """Fill missing abstracts through DOI (Crossref, metadata only)."""
    cr = (cfg.get("abstract_enrichment") or {}).get("crossref", {})
    if not cr.get("enabled"):
        return 0
    client = client or Client(cfg, log)
    rows = con.execute("""SELECT openalex_id, doi FROM papers WHERE COALESCE(abstract,'')='' AND doi!=''
                          AND crossref_checked=0""").fetchall()
    if limit:
        rows = rows[:limit]
    filled = 0
    for oid, doi in rows:
        try:
            d = client.get_json(f"{cr['base_url']}/{doi}")
            ab = _strip_jats(((d or {}).get("message") or {}).get("abstract", ""))
            if len(ab) > 50:
                con.execute("UPDATE papers SET abstract=?, abstract_source='crossref' WHERE openalex_id=?", (ab, oid))
                filled += 1
        except Exception as e:
            log_error(con, "crossref", doi, e)
        con.execute("UPDATE papers SET crossref_checked=1 WHERE openalex_id=?", (oid,))
        con.commit()
        time.sleep(float(cr.get("sleep_seconds", 0.1)))
    log.info("Crossref: %d abstracts filled out of %d checked", filled, len(rows))
    return filled


def enrich_semantic_scholar(con, cfg, log, client=None, limit=None):
    """Second abstract source (Semantic Scholar batch endpoint, by DOI). Optional; failures are non-fatal."""
    s2 = (cfg.get("abstract_enrichment") or {}).get("semantic_scholar", {})
    if not s2.get("enabled"):
        return 0
    client = client or Client(cfg, log)
    key = os.environ.get(s2.get("api_key_env", ""), "")
    headers = {"x-api-key": key} if key else None
    rows = con.execute("""SELECT openalex_id, doi FROM papers WHERE COALESCE(abstract,'')='' AND doi!=''
                          AND s2_checked=0""").fetchall()
    if limit:
        rows = rows[:limit]
    bs = int(s2.get("batch_size", 100))
    filled = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        try:
            res = client.post_json(s2["base_url"], json={"ids": [f"DOI:{d}" for _, d in chunk]},
                                   params={"fields": "abstract"}, headers=headers)
        except Exception as e:
            log.warning("Semantic Scholar batch failed (%s) - skipping the rest of this source", str(e)[:150])
            log_error(con, "semantic_scholar", i, e)
            break
        for (oid, _), item in zip(chunk, res or []):
            ab = clean_text((item or {}).get("abstract") or "")
            if len(ab) > 50:
                con.execute("UPDATE papers SET abstract=?, abstract_source='semanticscholar' WHERE openalex_id=?", (ab, oid))
                filled += 1
            con.execute("UPDATE papers SET s2_checked=1 WHERE openalex_id=?", (oid,))
        con.commit()
        time.sleep(float(s2.get("sleep_seconds", 1.2)))
    log.info("Semantic Scholar: %d abstracts filled out of %d checked", filled, len(rows))
    return filled


def estimate(cfg, log, client=None):
    """Dry run: ask OpenAlex how many records each query matches (1 request per query, nothing stored)."""
    client = client or Client(cfg, log)
    url = cfg["openalex"]["base_url"]
    rows, total = [], 0
    for q in cfg["queries"]:
        p = build_params(q, cfg, "*", client)
        p["per-page"] = 1
        p.pop("select", None)
        d = client.get_json(url, p) or {}
        n = (d.get("meta") or {}).get("count")
        rows.append((q["id"], n, q["q"]))
        total += n or 0
        print(f"{q['id']}: {n if n is not None else '?':>7}  {q['q'][:90]}")
    print(f"Sum over queries (upper bound, queries overlap heavily): {total:,}")
    return rows
