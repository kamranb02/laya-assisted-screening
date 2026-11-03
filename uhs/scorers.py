"""Turn raw Laya scores into the DEPLOYED decision score (raw, calibrated, stacked or TF-IDF baseline).

The scorer and threshold come from output/validation/operating_point.json, written by `python run.py evaluate`
from the human gold labels. Before validation exists, the raw Laya probability and
decision.raw_include_threshold are used and the report says so.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from .metrics import logit, sigmoid

OP_FILE = "operating_point.json"
RULE_ORDER = ["strong", "likely", "unlikely", "none"]


def load_operating_point(paths):
    f = paths.validation / OP_FILE
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def rule_dummies(rule_uhs):
    r = np.asarray(rule_uhs, object)
    return np.column_stack([(r == "strong"), (r == "likely"), (r == "unlikely")]).astype(float)


def stack_features(raw_p, rule_uhs):
    return np.column_stack([logit(raw_p), rule_dummies(rule_uhs)])


def tfidf_text(title, abstract):
    return (title.fillna("").astype(str) + ". " + abstract.fillna("").astype(str)).tolist()


def score_frame(df, op, paths, log=None):
    """Return (score array aligned with df, scorer name actually used)."""
    raw = df["uhs_prob"].to_numpy(float)
    if not op:
        return raw, "laya_raw"
    name = op.get("scorer", "laya_raw")
    try:
        if name == "laya_cal":
            a, b = op["calibration"]["a"], op["calibration"]["b"]
            return np.where(np.isnan(raw), np.nan, sigmoid(a * logit(np.nan_to_num(raw, nan=0.5)) + b)), name
        if name == "stack":
            c = np.array(op["stack"]["coef"], float)
            X = stack_features(np.nan_to_num(raw, nan=0.5), df["rule_uhs"].fillna("none"))
            return np.where(np.isnan(raw), np.nan, sigmoid(X @ c + op["stack"]["intercept"])), name
        if name == "tfidf":
            with open(paths.validation / "baseline_tfidf.pkl", "rb") as f:
                mdl = pickle.load(f)
            p = mdl.predict_proba(tfidf_text(df["title"], df["abstract"]))[:, 1]
            return np.where(np.isnan(raw), np.nan, p), name
    except Exception as e:  # never break the analysis because an artefact is missing
        if log:
            log.warning("Could not apply scorer %r (%s) - falling back to raw Laya scores", name, e)
    return raw, "laya_raw"
