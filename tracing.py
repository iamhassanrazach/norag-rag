"""Tracing (V4, part A): one TRACE per request (a question, an indexing run), one SPAN per step.

Our own implementation, no service to run: each finished trace is one JSON line in traces/YYYY-MM-DD.jsonl.
The vocabulary is OpenTelemetry's (trace, span, parent, attributes, status), so the same data could be sent to
Langfuse or Phoenix later; the LLM attributes use the gen_ai.* names where one exists.

    with tracing.trace("question", corpus="kestrel") as t:
        with tracing.span("retrieve.dense", k=20) as s:
            ...
            s.set(n_results=20)
        t.set(outcome="answered")

If no trace is open (the evaluation, the tests), span() does nothing: instrumenting a function costs nothing there.

What never goes in a trace: API keys (scrubbed at write time), whole documents (chunk ids only: the text is in the
index), personal data (pii.redact_text at write time). Traces older than TRACE_RETENTION_DAYS (30) are deleted.
"""
import contextlib
import contextvars
import datetime
import glob
import hashlib
import json
import os
import re
import threading
import time
import uuid

import pii

TRACE_DIR = os.getenv("TRACE_DIR", "traces")
RETENTION_DAYS = int(os.getenv("TRACE_RETENTION_DAYS", "30"))
_current = contextvars.ContextVar("current_trace", default=None)
_lock = threading.Lock()
_secrets = set()

_SECRET_PATTERNS = [re.compile(p) for p in (
    r"AIza[0-9A-Za-z_\-]{30,}",                                   # Google API keys
    r"\bsk-[A-Za-z0-9_\-]{16,}",                                  # OpenAI-style keys
    r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}",
    r"(?i)(api[_-]?key|token|secret|password)(['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._\-]{12,}",
)]


def add_secret(value):
    """Register a configured secret (the API key): it is removed from every trace, whatever its format."""
    if value and len(value) >= 8:
        _secrets.add(value)


def scrub(text):
    for s in _secrets:
        text = text.replace(s, "[SECRET]")
    for rx in _SECRET_PATTERNS:
        text = rx.sub(lambda m: "[SECRET]" if m.lastindex is None else f"{m.group(1)}{m.group(2)}[SECRET]", text)
    return text


def clean(value):
    """Applied to every attribute when the trace is written: secrets out, personal data out."""
    if isinstance(value, str):
        return pii.redact_text(scrub(value))
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def fingerprint(config):
    """Six characters that identify a configuration: same fingerprint = same pipeline."""
    return hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:6]


class _NullSpan:
    trace_id = None
    id = None

    def set(self, **_):
        pass

    def error(self, _):
        pass


class Span:
    def __init__(self, trace, name, parent, attrs):
        self.trace_id, self.id, self.name, self.parent = trace.id, uuid.uuid4().hex[:6], name, parent
        self.attrs, self.status = dict(attrs), "ok"
        self.t0 = time.perf_counter()
        self.start_ms = round((self.t0 - trace.t0) * 1000, 1)
        self.duration_ms = None

    def set(self, **attrs):
        self.attrs.update(attrs)

    def error(self, message):
        """Mark the span as failed without raising (a handled error: a 429 that we retry, a chunk without JSON...)."""
        self.status = "error"
        self.attrs["error"] = str(message)[:300]

    def record(self):
        return {"id": self.id, "parent": self.parent, "name": self.name, "start_ms": self.start_ms,
                "duration_ms": self.duration_ms, "status": self.status, "attrs": clean(self.attrs)}


class Trace:
    def __init__(self, kind, attrs):
        self.id = uuid.uuid4().hex[:8]
        self.trace_id = self.id
        self.kind, self.attrs = kind, dict(attrs)
        self.t0 = time.perf_counter()
        self.started_at = datetime.datetime.now(datetime.timezone.utc)
        self.spans, self.stack = [], []

    def set(self, **attrs):
        self.attrs.update(attrs)

    def error(self, message):
        self.attrs["outcome"], self.attrs["error"] = "error", str(message)[:300]

    def finish(self, failed):
        duration = round((time.perf_counter() - self.t0) * 1000, 1)
        llm = [s for s in self.spans if s.name == "llm.call"]
        totals = {
            "llm_calls": len(llm),
            "tokens_in": sum(s.attrs.get("gen_ai.usage.input_tokens", 0) for s in llm),
            "tokens_out": sum(s.attrs.get("gen_ai.usage.output_tokens", 0) for s in llm),
            "cost_usd": round(sum(s.attrs.get("cost_usd", 0.0) for s in llm), 6),
        }
        self.attrs.setdefault("outcome", "error" if failed else "ok")
        return {"trace_id": self.id, "kind": self.kind, "started_at": self.started_at.isoformat(timespec="seconds"),
                "duration_ms": duration, "outcome": self.attrs["outcome"], "attrs": clean(self.attrs),
                "totals": totals, "spans": [s.record() for s in self.spans]}


@contextlib.contextmanager
def trace(kind, **attrs):
    """Open a trace. Already inside one? Then this becomes a child span instead (an indexing run started from a request)."""
    cur = _current.get()
    if cur is not None:
        with span(kind, **attrs) as sp:
            yield sp
        return
    tr = Trace(kind, attrs)
    token = _current.set(tr)
    failed = False
    try:
        yield tr
    except BaseException as e:
        failed = True
        tr.attrs.setdefault("error", f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        _current.reset(token)
        _write(tr.finish(failed))


@contextlib.contextmanager
def span(name, **attrs):
    tr = _current.get()
    if tr is None:
        yield _NullSpan()
        return
    sp = Span(tr, name, tr.stack[-1].id if tr.stack else None, attrs)
    tr.spans.append(sp)
    tr.stack.append(sp)
    try:
        yield sp
    except BaseException as e:
        sp.status = "error"
        sp.attrs.setdefault("error", f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        sp.duration_ms = round((time.perf_counter() - sp.t0) * 1000, 1)
        tr.stack.pop()


def current_id():
    tr = _current.get()
    return tr.id if tr else None


# ------------------------------------------------------------------ storage

def _file(day):
    return os.path.join(TRACE_DIR, f"{day}.jsonl")


def _write(record):
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        day = record["started_at"][:10]
        with _lock, open(_file(day), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        purge_old()
    except OSError:
        pass  # a trace that cannot be written must never break the request it describes


def purge_old(days=None):
    """Retention: delete the daily files older than `days` days. Returns how many were deleted."""
    days = RETENTION_DAYS if days is None else days
    limit = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).date().isoformat()
    n = 0
    for path in glob.glob(os.path.join(TRACE_DIR, "*.jsonl")):
        if os.path.basename(path)[:10] < limit:
            try:
                os.remove(path)
                n += 1
            except OSError:
                pass
    return n


def _records():
    for path in sorted(glob.glob(os.path.join(TRACE_DIR, "*.jsonl")), reverse=True):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                yield json.loads(line)
            except ValueError:
                continue


def get_trace(trace_id):
    return next((r for r in _records() if r["trace_id"] == trace_id), None)


def list_traces(limit=50, kind=None):
    out = []
    for r in _records():
        if kind and r["kind"] != kind:
            continue
        out.append({k: r[k] for k in ("trace_id", "kind", "started_at", "duration_ms", "outcome", "totals")}
                   | {"title": r["attrs"].get("question") or r["attrs"].get("corpus") or "",
                      "corpus": r["attrs"].get("corpus", "")})
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ metrics, computed from the traces

def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, -(-len(values) * p // 100) - 1))]   # nearest rank


def metrics(days=7):
    """Volume and outcomes, latency p50/p95 (whole request and each step), tokens and cost per day, from the traces."""
    limit = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
    questions, per_step, outcomes, per_day, index_runs = [], {}, {}, {}, 0
    for r in _records():
        if r["started_at"] < limit:
            break
        day = per_day.setdefault(r["started_at"][:10], {"requests": 0, "tokens": 0, "cost_usd": 0.0})
        day["requests"] += 1
        day["tokens"] += r["totals"]["tokens_in"] + r["totals"]["tokens_out"]
        day["cost_usd"] = round(day["cost_usd"] + r["totals"]["cost_usd"], 6)
        if r["kind"] == "index":
            index_runs += 1
        if r["kind"] != "question":
            continue
        questions.append(r["duration_ms"])
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        for s in r["spans"]:
            per_step.setdefault(s["name"], []).append(s["duration_ms"] or 0)
    return {"days": days, "questions": len(questions), "outcomes": outcomes, "index_runs": index_runs,
            "latency_ms": {"p50": _pct(questions, 50), "p95": _pct(questions, 95)},
            "steps": {n: {"p50": _pct(v, 50), "p95": _pct(v, 95), "n": len(v)} for n, v in sorted(per_step.items())},
            "per_day": dict(sorted(per_day.items()))}
