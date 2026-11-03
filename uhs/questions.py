"""Question set sent to Laya (single source of truth).

Questions are defined in a neutral internal format and converted per `screening.question_style`:
  binary  -> Laya `noul`  (style "noul")          or a neutral A/B `choice` (style "choice_ab")
  choice  -> Laya `choice` (unchanged)
Answers are canonicalised back to {"type": "noul", "noul": P(yes)} / {"type": "choice", ...} so the rest of the
pipeline never depends on the style. CHANGING ANY TEXT BELOW CHANGES `question_hash()`; `screen` then refuses
to mix old and new answers in one database (use `screen --rescreen`).

Budget note: Laya reserves `head_max_len` tokens (192 for the English checkpoint) for the question and its options.
Keep instructions short; every option should be a short phrase.
"""
from __future__ import annotations

import hashlib
import json


def B(instructions, yes="yes", no="no"):
    return {"type": "binary", "instructions": instructions, "yes": yes, "no": no}


def C(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


Q_UHS = {"uhs": B(
    "Is this paper substantially about underground hydrogen storage (salt caverns, depleted reservoirs, aquifers, "
    "rock caverns), including its rock-fluid, geochemical, microbial, geomechanical or economic aspects? "
    "Answer no for hydrogen production, fuel cells, tanks, hydrides, or CO2-only or natural-gas-only storage.",
    yes="yes, mainly about underground hydrogen storage", no="no, a different topic")}

Q_DETAIL = {
    "storage_type": C("What is the principal underground storage setting?", {
        "depleted_gas_reservoir": "depleted gas, oil or hydrocarbon reservoir",
        "aquifer": "saline aquifer or porous formation",
        "salt_cavern": "salt cavern or salt dome",
        "hard_rock_cavern": "lined or unlined rock cavern, mine",
        "other_geology": "basalt, coal seam or other formation",
        "general_or_unclear": "general, several settings, or unclear"}),
    "method": C("What is the main research method?", {
        "experimental": "laboratory or experimental work",
        "numerical": "numerical or computational modelling",
        "analytical": "analytical or theoretical model",
        "data_driven": "machine learning or statistical data analysis",
        "techno_economic": "techno-economic, policy or risk assessment",
        "review": "review, perspective or bibliometric"}),
    "scale": C("What is the main scale of investigation?", {
        "pore": "pore scale or molecular scale",
        "core": "core, sample or laboratory scale",
        "reservoir": "reservoir or well scale",
        "field": "field, project, regional or system scale",
        "other": "unclear or not applicable"}),
    "multiphase_flow": B("Does it substantially study multiphase flow, capillary pressure, relative permeability, "
                         "wettability or displacement relevant to hydrogen storage?"),
    "geochemistry": B("Does it substantially study geochemistry, mineral reactions or reactive transport "
                      "relevant to hydrogen storage?"),
    "biology_methanogenesis": B("Does it substantially study microorganisms, microbial reactions or "
                                "methanogenesis during hydrogen storage?"),
    "geomechanics_integrity": B("Does it substantially study geomechanics, caprock or well integrity, "
                                "or leakage relevant to hydrogen storage?"),
    "experimental": B("Does it contain substantial experimental or laboratory work?"),
    "numerical": B("Does it contain substantial numerical or computational modeling?"),
    "data_driven_ml": B("Does it substantially use machine learning, surrogate models or data-driven methods?"),
}
QUESTIONS_ALL = {**Q_UHS, **Q_DETAIL}
BINARY_DETAIL = [q for q, d in Q_DETAIL.items() if d["type"] == "binary"]
CHOICE_DETAIL = [q for q, d in Q_DETAIL.items() if d["type"] == "choice"]


def to_laya(qdefs: dict, style: str) -> dict:
    """Internal definitions -> the question dict understood by laya.predict / predict_batch."""
    out = {}
    for q, d in qdefs.items():
        if d["type"] == "choice":
            out[q] = {"type": "choice", "instructions": d["instructions"], "criteria": dict(d["criteria"])}
        elif style == "noul":
            out[q] = {"type": "noul", "instructions": d["instructions"]}
        elif style == "choice_ab":
            out[q] = {"type": "choice", "instructions": d["instructions"],
                      "criteria": {"A": d["yes"], "B": d["no"]}}
        else:
            raise ValueError(f"unknown question_style {style!r} (use choice_ab or noul)")
    return out


def question_hash(style: str) -> str:
    blob = json.dumps({"q": QUESTIONS_ALL, "style": style}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def canonicalise(answers: dict, qdefs: dict, style: str) -> dict:
    """Raw Laya answers -> uniform schema. Binary questions always become {'type':'noul','noul':P(yes)}."""
    out = {}
    for q, d in qdefs.items():
        a = (answers or {}).get(q)
        if a is None:
            continue
        if d["type"] == "binary":
            if style == "noul":
                out[q] = {"type": "noul", "noul": float(a["noul"])}
            else:
                probs = {str(k).upper(): float(v) for k, v in (a.get("probabilities") or {}).items()}
                if "A" in probs:
                    tot = sum(v for k, v in probs.items() if k in ("A", "B")) or 1.0
                    p = probs["A"] / tot
                else:  # model returned no distribution: fall back to the hard choice (flagged by p in {0,1})
                    p = 1.0 if str(a.get("choice", "")).upper() == "A" else 0.0
                out[q] = {"type": "noul", "noul": float(p)}
        else:
            out[q] = {"type": "choice", "choice": a.get("choice"),
                      "probabilities": {k: float(v) for k, v in (a.get("probabilities") or {}).items()}}
    return out
