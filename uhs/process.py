"""Stage 2 - de-duplication, eligibility rules and rule-based tagging (no ML, fully deterministic)."""
from __future__ import annotations

from .common import norm_title
from .lexicon import tag_record


def dedup(con, log) -> int:
    """Flag duplicates (same DOI, or same normalised title) - nothing is deleted, only `dup_of` is set.

    OpenAlex often holds a preprint and the journal version as separate works; the journal version wins.
    """
    rows = con.execute("SELECT openalex_id,doi,title,type,cited_by_count,abstract FROM papers").fetchall()
    parent = {r[0]: r[0] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen = {}
    for oid, doi, title, *_ in rows:
        keys = []
        if doi:
            keys.append(("d", doi))
        tk = norm_title(title)
        if len(tk) >= 25:
            keys.append(("t", tk))
        for k in keys:
            if k in seen:
                union(seen[k], oid)
            else:
                seen[k] = oid
    groups = {}
    for r in rows:
        groups.setdefault(find(r[0]), []).append(r)
    con.execute("UPDATE papers SET dup_of=NULL")
    n = 0
    for members in groups.values():
        if len(members) == 1:
            continue
        members.sort(key=lambda r: (r[3] == "preprint", not r[1], not (r[5] or ""), -(r[4] or 0), r[0]))
        rep = members[0][0]
        for m in members[1:]:
            con.execute("UPDATE papers SET dup_of=? WHERE openalex_id=?", (rep, m[0]))
            n += 1
    con.commit()
    log.info("Dedup: %d duplicate records flagged", n)
    return n


def apply_rules(con, cfg, log):
    """Eligibility (type / year / retraction / no-hydrogen) + lexicon tags for every record."""
    y0, y1 = cfg["_start_year"], cfg["_end_year"]
    bad_types = set(cfg["screening"].get("exclude_types", []))
    rows = con.execute("SELECT openalex_id,title,abstract,type,publication_year,is_retracted FROM papers").fetchall()
    stats = {}
    for oid, title, abs_, typ, yr, retr in rows:
        t = tag_record(title, abs_)
        elig, reason = "eligible", None
        if typ in bad_types:
            elig, reason = "auto_excluded", f"type:{typ}"
        elif retr:
            elig, reason = "auto_excluded", "retracted"
        elif yr is None or not (y0 <= int(yr) <= y1):
            elig, reason = "auto_excluded", "year_out_of_range"
        elif not t["has_hydrogen"] and (abs_ or "").strip():
            elig, reason = "auto_excluded", "no_hydrogen_term"
        # NOTE: a record WITHOUT abstract is never auto-excluded here (title alone cannot prove absence of
        # hydrogen); it is screened on the title and flagged `abstract_missing` for the review queue.
        stats[reason or "eligible"] = stats.get(reason or "eligible", 0) + 1
        con.execute("""UPDATE papers SET eligibility=?, exclude_reason=?, tags=?, tools=?, rule_uhs=?,
                       rule_storage=?, rule_scale=?, rule_method=? WHERE openalex_id=?""",
                    (elig, reason, ";".join(t["tags"]), ";".join(t["tools"]), t["rule_uhs"],
                     t["rule_storage"], t["rule_scale"], t["rule_method"], oid))
    con.commit()
    log.info("Eligibility summary: %s", stats)
    return stats


def process(con, cfg, log):
    dedup(con, log)
    return apply_rules(con, cfg, log)
