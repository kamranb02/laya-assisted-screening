"""Version-tolerant access to Laya.

Why this module exists
  * Laya ships three checkpoints (english root / multilingual / typed-decisions) with different context budgets
    (512 / 1024 / 1024 tokens, of which `head_max_len` is reserved for the question) and the runtime changes between
    releases (0.3.20 at the time of writing). Everything that could change is isolated here and recorded as
    provenance, so that the rest of the pipeline never has to know.
  * The API is probed instead of assumed: `predict_batch` is used when present, otherwise `predict` per record;
    optional keyword arguments are passed only when the installed version accepts them.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import re

from .questions import canonicalise, question_hash, to_laya

os.environ.setdefault("USE_TF", "0")        # model card: TF/abseil can deadlock model construction

SUBFOLDER = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
DEFAULT_CTX = {"english": (512, 192), "multilingual": (1024, 256), "typed-decisions": (1024, 256), "custom": (512, 192)}
CHARS_PER_TOKEN = 4.0                       # conservative for scientific English


def est_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


# ----------------------------------------------------------------- state building
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def compose_state(title: str, abstract: str, mode: str, budget_tokens: int, max_chars: int):
    """Return (state_dict, trimmed_flag, est_tokens_of_state)."""
    title = title or ""
    ab = (abstract or "")[:max_chars]
    if mode == "full" or not ab:
        return {"title": title, "abstract": ab}, 0, est_tokens(title + " " + ab)
    room = max(40, budget_tokens - est_tokens(title) - 8)
    if est_tokens(ab) <= room:
        return {"title": title, "abstract": ab}, 0, est_tokens(title + " " + ab)
    if mode == "head":
        cut = ab[: int(room * CHARS_PER_TOKEN)]
        return {"title": title, "abstract": cut}, 1, est_tokens(title + " " + cut)
    # smart_trim: first sentences (~65 %) + last sentences (~30 %) of the token room
    sents = _SENT.split(ab)
    head_c, tail_c = int(room * CHARS_PER_TOKEN * 0.65), int(room * CHARS_PER_TOKEN * 0.30)
    head, used = [], 0
    for s in sents:
        if used + len(s) > head_c and head:
            break
        head.append(s)
        used += len(s) + 1
    tail, used_t = [], 0
    for s in reversed(sents[len(head):]):
        if used_t + len(s) > tail_c:
            break
        tail.insert(0, s)
        used_t += len(s) + 1
    text = " ".join(head) + (" [...] " + " ".join(tail) if tail else "")
    if len(text) > room * CHARS_PER_TOKEN:          # a single huge sentence: hard cut
        text = text[: int(room * CHARS_PER_TOKEN)]
    return {"title": title, "abstract": text}, 1, est_tokens(title + " " + text)


# ----------------------------------------------------------------------- agent
def installed_versions() -> dict:
    v = {}
    for m in ("laya", "torch", "transformers", "huggingface_hub"):
        try:
            mod = importlib.import_module(m)
            ver = getattr(mod, "__version__", None)
            if not ver:
                from importlib.metadata import version
                ver = version(m)
            v[m] = str(ver)
        except Exception as e:
            v[m] = f"unavailable ({type(e).__name__})"
    return v


def hub_revision(repo: str):
    """Best-effort commit hash of the model repo (needs network; returns None offline)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo).sha
    except Exception:
        return None


class LayaAgent:
    """Thin wrapper around a loaded laya agent + the settings that matter for reproducibility."""

    def __init__(self, raw, checkpoint, model_id, style, max_len=0, head_max_len=0, device="cpu", batch_size=8):
        self.raw, self.checkpoint, self.model_id, self.style = raw, checkpoint, model_id, style
        self.device = str(getattr(raw, "device", device))
        dm, dh = DEFAULT_CTX.get(checkpoint, DEFAULT_CTX["custom"])
        cfgd = getattr(raw, "cfg", None)
        if isinstance(cfgd, dict):
            dm = int(cfgd.get("max_len", dm) or dm)
            dh = int(cfgd.get("head_max_len", dh) or dh)
            if head_max_len:
                cfgd["head_max_len"] = int(head_max_len)
        self.max_len = int(max_len or dm)
        self.head_max_len = int(head_max_len or dh)
        self.batch_size = batch_size
        self.text_budget = max(64, self.max_len - self.head_max_len)
        self._pb = getattr(raw, "predict_batch", None)
        self._p = getattr(raw, "predict", None)
        if not (self._pb or self._p):
            raise RuntimeError("The loaded Laya object has neither predict_batch nor predict - "
                               "incompatible Laya version; see laya_compat.py")

    @staticmethod
    def _accepts(fn, name):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return name in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    def raw_predict(self, states, laya_questions, batch_size=None):
        bs = batch_size or self.batch_size
        if self._pb:
            kw = {}
            if self._accepts(self._pb, "batch_size"):
                kw["batch_size"] = bs
            if self._accepts(self._pb, "max_len"):
                kw["max_len"] = self.max_len
            return self._pb(states, laya_questions, **kw)
        kw = {"max_len": self.max_len} if self._accepts(self._p, "max_len") else {}
        return [self._p(s, laya_questions, **kw) for s in states]

    def predict(self, states, qdefs, batch_size=None):
        """-> list of {'answers': canonical answers, 'tokens': int}"""
        res = self.raw_predict(states, to_laya(qdefs, self.style), batch_size)
        out = []
        for r in res:
            out.append({"answers": canonicalise(r.get("answers"), qdefs, self.style),
                        "tokens": int((r.get("usage") or {}).get("input_tokens", 0))})
        return out

    def with_style(self, style):
        """Same loaded weights, different question style (no second model load)."""
        a = object.__new__(LayaAgent)
        a.__dict__.update(self.__dict__)
        a.style = style
        return a

    def provenance(self) -> dict:
        d = installed_versions()
        d.update(checkpoint=self.checkpoint, model_id=self.model_id, device=self.device, max_len=self.max_len,
                 head_max_len=self.head_max_len, question_style=self.style, question_hash=question_hash(self.style))
        return d

    def tag(self) -> str:
        v = installed_versions().get("laya", "?")
        return f"laya-{v}|{self.checkpoint}|{self.style}|q:{question_hash(self.style)}|len{self.max_len}"


def _resolve_device(sc, log):
    import torch
    dev = sc.get("device", "cpu")
    if dev == "auto":
        dev = "cpu"
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_properties(0).total_memory >= 4 * 1024 ** 3:
                    dev = "cuda"
            except Exception:
                pass
    threads = int(sc.get("threads") or 0) or (os.cpu_count() or 4)
    torch.set_num_threads(threads)
    return dev, threads


def load_agent(sc: dict, log, checkpoint=None, style=None, max_len=None, model_id=None, head_max_len=None) -> LayaAgent:
    """Load one Laya checkpoint according to the `screening` config (overridable per call)."""
    try:
        import laya
    except ImportError as e:
        raise RuntimeError("Laya is not installed in this Python environment: run `pip install -U laya` "
                           "(Python >= 3.10), then `python run.py doctor`.") from e
    checkpoint = checkpoint or sc.get("checkpoint", "english")
    if checkpoint == "auto":
        checkpoint = "english"
    model_id = model_id or sc.get("model_id", "convaiinnovations/laya")
    style = style or sc.get("question_style", "choice_ab")
    dev, threads = _resolve_device(sc, log)
    sub = SUBFOLDER.get(checkpoint)
    log.info("Loading Laya %s: repo=%s subfolder=%s device=%s threads=%d (laya %s)", checkpoint, model_id, sub, dev,
             threads, installed_versions().get("laya"))
    kw = {"device": dev}
    if sub:
        kw["subfolder"] = sub
    raw = laya.load(model_id, **kw)
    return LayaAgent(raw, checkpoint, model_id, style,
                     max_len=int(max_len if max_len is not None else sc.get("max_len") or 0),
                     head_max_len=int(head_max_len if head_max_len is not None else sc.get("head_max_len") or 0),
                     device=dev, batch_size=int(sc.get("batch_size", 8)))


class AgentPool:
    """Gives the right agent per record. `auto` routes by OpenAlex language (English vs the rest)."""

    def __init__(self, sc, log, agents=None):
        self.sc, self.log = sc, log
        self.mode = sc.get("checkpoint", "english")
        self._agents = dict(agents or {})

    def route(self, language) -> str:
        if self.mode == "auto":
            return "english" if (not language or str(language).lower().startswith("en")) else "multilingual"
        return self.mode

    def get(self, route) -> LayaAgent:
        if route not in self._agents:
            self._agents[route] = load_agent(self.sc, self.log, checkpoint=route)
        return self._agents[route]

    def routes(self):
        return ["english", "multilingual"] if self.mode == "auto" else [self.mode]

    def tag(self) -> str:
        """Tag of the pool; for auto mode it combines both checkpoints but does not need them loaded."""
        v = installed_versions().get("laya", "?")
        sc = self.sc
        style = sc.get("question_style", "choice_ab")
        ml = sc.get("max_len") or 0
        return f"laya-{v}|{self.mode}|{style}|q:{question_hash(style)}|len{ml or 'default'}"

    def provenance(self) -> dict:
        d = installed_versions()
        d.update(checkpoint_mode=self.mode, model_id=self.sc.get("model_id"), question_style=self.sc.get("question_style"),
                 question_hash=question_hash(self.sc.get("question_style", "choice_ab")),
                 revision=hub_revision(self.sc.get("model_id", "")), max_len_cfg=self.sc.get("max_len"),
                 head_max_len_cfg=self.sc.get("head_max_len"), state_mode=self.sc.get("state_mode"))
        for r, a in self._agents.items():
            d[f"loaded_{r}"] = {"max_len": a.max_len, "head_max_len": a.head_max_len, "device": a.device}
        return d


def probe(agent: LayaAgent) -> dict:
    """Schema probe used by `doctor`/`smoke`: one tiny call, checks that the answer layout is understood."""
    from .questions import Q_UHS
    st = [{"title": "Hydrogen storage in salt caverns", "abstract": "Cyclic injection in salt caverns for underground hydrogen storage."}]
    try:
        r = agent.predict(st, Q_UHS)
        p = r[0]["answers"]["uhs"]["noul"]
        return {"ok": 0.0 <= p <= 1.0, "p_yes": p}
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
