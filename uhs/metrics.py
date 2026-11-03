"""Small, dependency-light statistics used by validation (weighted for stratified sampling)."""
from __future__ import annotations

import math

import numpy as np


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_platt(s, y, w=None, lam=1e-3, iters=60):
    """Weighted logistic recalibration  P = sigmoid(a*logit(s) + b)  (Newton-Raphson, tiny ridge toward a=1,b=0)."""
    x = logit(s)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    w = w / w.mean()
    X = np.column_stack([x, np.ones(len(x))])
    th = np.array([1.0, 0.0])
    prior = th.copy()
    for _ in range(iters):
        p = sigmoid(X @ th)
        g = X.T @ (w * (p - y)) + lam * (th - prior)
        H = (X * (w * p * (1 - p))[:, None]).T @ X + lam * np.eye(2)
        step = np.linalg.solve(H, g)
        th = th - step
        if np.abs(step).max() < 1e-8:
            break
    return float(th[0]), float(th[1])


def apply_platt(s, a, b):
    return sigmoid(a * logit(s) + b)


def wconf(y, pred, w=None):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return dict(tp=float(w[(y == 1) & (pred == 1)].sum()), fp=float(w[(y == 0) & (pred == 1)].sum()),
                tn=float(w[(y == 0) & (pred == 0)].sum()), fn=float(w[(y == 1) & (pred == 0)].sum()))


def prf(c):
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else np.nan
    n = tp + fp + tn + fn
    return dict(precision=prec, recall=rec, specificity=spec, f1=f1, accuracy=(tp + tn) / n if n else np.nan)


def auc_w(y, s, w=None):
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return np.nan
    sp, sn, wp, wn = s[pos], s[neg], w[pos], w[neg]
    gt = (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :])
    return float((wp[:, None] * wn[None, :] * gt).sum() / (wp.sum() * wn.sum()))


def ece_w(y, p, w=None, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot, e = w.sum(), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += w[m].sum() / tot * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return float(e)


def brier_w(y, p, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float(np.average((p - y) ** 2, weights=w))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def kappa(a, b, w=None):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    po = float(w[a == b].sum())
    pe = sum(float(w[a == k].sum()) * float(w[b == k].sum()) for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def macro_f1(y, p, labels, w=None):
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    f = []
    for k in labels:
        tp = float(w[(y == k) & (p == k)].sum())
        fp = float(w[(y != k) & (p == k)].sum())
        fn = float(w[(y == k) & (p != k)].sum())
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else np.nan


def stratified_folds(strata, y, k, seed):
    """Fold id per row, balanced within (stratum, label) groups."""
    rng = np.random.default_rng(seed)
    key = np.array([f"{s}|{int(v)}" for s, v in zip(strata, y)])
    folds = np.zeros(len(key), int)
    for g in np.unique(key):
        idx = np.where(key == g)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def bootstrap(fn, strata, reps, seed):
    """Percentile 95% CI of fn(index_array) with resampling WITHIN strata (keeps the sampling design)."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    out = []
    for _ in range(int(reps)):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        out.append(fn(idx))
    arr = np.array(out, float)
    return (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)))


def two_prop_p(k1, n1, k2, n2):
    """Two-sided p-value of a two-proportion z-test."""
    if min(n1, n2) == 0:
        return np.nan
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return float(math.erfc(z / math.sqrt(2)))
