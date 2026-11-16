"""V4: traces, personal-data redaction, LLM_URL / LLM_API_KEY / LLM_MODEL, uploads, indexing jobs, concept map explorer.
No network and no real LLM: a fake OpenAI-compatible server (tests/fake_llm.py) plays the provider."""
import datetime
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

import graph
import llm
import pii
import rag
import tracing
from app import app
from fake_llm import FakeLLM

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing, "TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr(rag, "USER_CORPORA_DIR", str(tmp_path / "user"))
    monkeypatch.setattr(graph, "GRAPH_DIR", str(tmp_path / "graph"))
    monkeypatch.delenv("PII_NAMES", raising=False)


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setenv("LLM_URL", f.url)
    monkeypatch.setenv("LLM_API_KEY", "test-key-123456")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    yield f
    f.stop()


def trace_of(response):
    r = client.get("/api/traces/" + response.json()["trace_id"])
    assert r.status_code == 200
    return r.json()


# ------------------------------------------------------------------------------------------------ tracing

def test_trace_is_a_tree_of_spans_written_to_a_file():
    with tracing.trace("question", corpus="k", question="hello") as t:
        with tracing.span("retrieve", k=5):
            with tracing.span("retrieve.dense") as s:
                s.set(n_results=3)
        with pytest.raises(ValueError):
            with tracing.span("generate"):
                raise ValueError("boom")
        t.set(outcome="answered")
    rec = tracing.get_trace(t.trace_id)
    names = {s["name"]: s for s in rec["spans"]}
    assert names["retrieve.dense"]["parent"] == names["retrieve"]["id"] and names["retrieve"]["parent"] is None
    assert names["retrieve.dense"]["attrs"]["n_results"] == 3
    assert names["generate"]["status"] == "error" and "boom" in names["generate"]["attrs"]["error"]
    assert rec["outcome"] == "answered" and rec["duration_ms"] >= 0
    assert os.listdir(tracing.TRACE_DIR)[0].endswith(".jsonl")


def test_span_without_a_trace_does_nothing():
    with tracing.span("x", a=1) as s:
        s.set(b=2)
    assert tracing.current_id() is None


def test_trace_never_contains_secrets_or_personal_data():
    tracing.add_secret("my-very-secret-key-value")
    with tracing.trace("question", question="mail jane@x.org please") as t:
        with tracing.span("llm.call", note="key my-very-secret-key-value and AIzaSyA1234567890abcdefghijklmnopqrstuv") as s:
            s.set(header="Authorization: Bearer abcdefghijklmnop1234")
    raw = open(os.path.join(tracing.TRACE_DIR, os.listdir(tracing.TRACE_DIR)[0]), encoding="utf-8").read()
    for leaked in ("my-very-secret-key-value", "AIzaSy", "abcdefghijklmnop1234", "jane@x.org"):
        assert leaked not in raw
    assert "[EMAIL]" in raw and "[SECRET]" in raw


def test_retention_deletes_old_files_only():
    os.makedirs(tracing.TRACE_DIR, exist_ok=True)
    old = (datetime.date.today() - datetime.timedelta(days=45)).isoformat()
    open(os.path.join(tracing.TRACE_DIR, f"{old}.jsonl"), "w").write("{}\n")
    with tracing.trace("question"):
        pass
    assert f"{old}.jsonl" not in os.listdir(tracing.TRACE_DIR) and len(os.listdir(tracing.TRACE_DIR)) == 1


def test_percentiles_use_the_nearest_rank():
    assert tracing._pct([10, 20, 30, 40], 50) == 20 and tracing._pct([10, 20, 30, 40], 95) == 40 and tracing._pct([], 50) is None


# ------------------------------------------------------------------------------------------------ personal data

def test_pii_detects_checks_and_leaves_identifiers_alone():
    r = pii.Redactor(True)
    text = ("Mail jane.doe@x.org, call +33 6 12 34 56 78 or 06 12 34 56 78. Card 4539 1488 0343 6467. "
            "IBAN FR14 2004 1010 0505 0001 3M02 606. Ask Dr. Anna Lopez.")
    out = r.redact(text)
    assert not any(v in out for v in ("jane.doe@x.org", "56 78", "4539", "FR14", "Anna Lopez"))
    assert r.summary() == {"EMAIL": 1, "PHONE": 2, "CARD": 1, "IBAN": 1, "NAME": 1}
    keep = "Error KST-2044 from 203.0.113.10, version 3.5.1, order 4539 1488 0343 6468, date 2026-11-15, 180 PS."
    assert pii.Redactor(True).redact(keep) == keep      # not a valid card (Luhn), not a phone, not a name


def test_pii_restore_puts_the_real_values_back():
    r = pii.Redactor(True)
    safe = r.redact("write to a@b.org or c@d.org, again a@b.org")
    assert safe == "write to [EMAIL_1] or [EMAIL_2], again [EMAIL_1]"
    assert r.restore("Use [EMAIL_2].") == "Use c@d.org."


def test_pii_known_names_and_switch_off(monkeypatch):
    monkeypatch.setenv("PII_NAMES", "Hassan Raza")
    assert pii.Redactor(True).redact("I am hassan raza") == "I am [NAME_1]"
    monkeypatch.setenv("PII_REDACTION", "0")
    assert pii.Redactor().redact("a@b.org") == "a@b.org"


# ------------------------------------------------------------------------------------------------ LLM_URL / LLM_API_KEY / LLM_MODEL

def test_llm_uses_the_url_key_and_model_from_the_environment(fake):
    with tracing.trace("question") as t:
        assert llm._complete("hello") == "ok"
    assert fake.requests[0]["auth"] == "Bearer test-key-123456" and fake.requests[0]["model"] == "my-model"
    span = tracing.get_trace(t.trace_id)["spans"][0]
    assert span["name"] == "llm.call" and span["attrs"]["gen_ai.request.model"] == "my-model"
    assert span["attrs"]["gen_ai.response.model"] == "fake-model-served" and span["attrs"]["gen_ai.usage.input_tokens"] >= 1


def test_old_gemini_settings_still_work(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "old-gemini-key-1")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-old")
    assert llm._keys() == ["old-gemini-key-1"] and llm.model() == "gemini-old" and "googleapis" in llm.url()


def test_no_key_gives_a_clear_error():
    with pytest.raises(llm.LLMError, match="LLM_API_KEY"):
        llm._keys()


def test_local_server_without_key_is_allowed(monkeypatch):
    monkeypatch.setenv("LLM_URL", "http://localhost:11434/v1")
    assert llm._keys() == ["none"]


# ------------------------------------------------------------------------------------------------ one trace per question

def ask(question, **kw):
    return client.post("/api/ask", json={"corpus": "kestrel", "question": question, "retrievers": ["dense", "bm25"], **kw})


def test_an_answer_comes_with_its_trace(fake):
    r = ask("What does error KST-2044 mean?")
    assert r.status_code == 200 and r.json()["answer"] and r.json()["in_documents"]
    t = trace_of(r)
    names = [s["name"] for s in t["spans"]]
    for step in ("retrieve", "retrieve.dense", "retrieve.bm25", "fusion", "prompt.build", "llm.answer", "llm.call", "validate"):
        assert step in names
    assert t["outcome"] == "answered" and t["kind"] == "question" and len(t["attrs"]["fingerprint"]) == 6
    assert t["totals"]["llm_calls"] == 1 and t["totals"]["tokens_in"] > 0 and t["totals"]["cost_usd"] > 0
    assert t["attrs"]["config"]["llm"] == "my-model" and t["attrs"]["config"]["corpus"].startswith("kestrel@")
    assert r.json()["summary"]["tokens_in"] == t["totals"]["tokens_in"]
    fused = next(s for s in t["spans"] if s["name"] == "fusion")["attrs"]["fused"]
    assert set(fused[0]["found_by"]) <= {"dense", "bm25"}                    # which retriever found which chunk, at which rank
    assert "kestrel_" not in open(os.path.join(tracing.TRACE_DIR, os.listdir(tracing.TRACE_DIR)[0]), encoding="utf-8").read().replace("kestrel@", "")  # (no raw index names)


def test_personal_data_never_reaches_the_provider_or_the_trace_but_the_user_still_sees_it(fake):
    r = ask("Refund requests must be sent by the workspace Owner. Write to jane@x.org please")
    assert r.status_code == 200
    assert any("kestrel.example" in c["text"] for c in r.json()["chunks"])            # a chunk that holds an e-mail address was in the prompt
    prompts = " ".join(x["prompt"] for x in fake.requests)
    assert "jane@x.org" not in prompts and "kestrel.example" not in prompts and "[EMAIL_1]" in prompts
    answer = r.json()["answer"]
    assert "jane@x.org" in answer and "billing@kestrel.example" in answer and "[EMAIL_" not in answer   # restored locally
    assert r.json()["privacy"]["EMAIL"] >= 2
    raw = open(os.path.join(tracing.TRACE_DIR, os.listdir(tracing.TRACE_DIR)[0]), encoding="utf-8").read()
    assert "jane@x.org" not in raw and "kestrel.example" not in raw


def test_invalid_model_output_is_an_outcome_with_a_trace(fake):
    r = ask("INVALIDJSON what is KST-2044?")
    assert r.status_code == 502 and "invalid" in r.json()["detail"].lower()
    t = client.get("/api/traces/" + r.json()["trace_id"]).json()
    assert t["outcome"] == "invalid_output" and next(s for s in t["spans"] if s["name"] == "validate")["status"] == "error"


def test_abstention_is_an_outcome(fake):
    r = ask("UNANSWERABLE: what is the CEO's favourite colour?")
    assert r.json()["in_documents"] is False and trace_of(r)["outcome"] == "abstained"


def test_a_failure_before_the_llm_still_has_a_trace():
    r = client.post("/api/ask", json={"corpus": "kestrel", "question": "who owns it?", "retrievers": ["graph"]})
    assert r.status_code == 400 and "concept graph" in r.json()["detail"]
    t = client.get("/api/traces/" + r.json()["trace_id"]).json()
    assert t["outcome"] == "error"


def test_traces_list_and_metrics_come_from_the_trace_files(fake):
    ask("What does error KST-2044 mean?")
    ask("UNANSWERABLE x")
    listing = client.get("/api/traces").json()
    assert [x["kind"] for x in listing] == ["question", "question"] and listing[0]["title"].startswith("UNANSWERABLE")
    m = client.get("/api/metrics").json()
    assert m["questions"] == 2 and m["outcomes"] == {"answered": 1, "abstained": 1} and m["latency_ms"]["p95"] >= m["latency_ms"]["p50"]
    assert "llm.call" in m["steps"] and list(m["per_day"].values())[0]["tokens"] > 0
    assert client.get("/api/traces/nope").status_code == 404


# ------------------------------------------------------------------------------------------------ upload + index + graph

CARS = ("# Lamborghini notes\n\nThe Revuelto replaced the Aventador in 2023 as the brand's flagship model.\n\n"
        "The Aventador is built by Lamborghini in Sant'Agata Bolognese and uses a naturally aspirated V12 engine.\n\n"
        "The Huracan is the smaller V10 model sold next to the Aventador.\n")


def make_cars(name="cars"):
    assert client.post("/api/corpora", json={"name": name}).status_code == 200
    assert client.put(f"/api/corpora/{name}/documents", json={"filename": "notes.md", "content": CARS}).status_code == 200


def wait_job(job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get("/api/jobs/" + job_id).json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("the job never finished")


def test_upload_rules():
    assert client.post("/api/corpora", json={"name": "Bad Name!"}).status_code == 400
    make_cars()
    assert client.post("/api/corpora", json={"name": "cars"}).status_code == 400                      # already exists
    put = lambda f, c: client.put("/api/corpora/cars/documents", json={"filename": f, "content": c})
    assert put("virus.exe", "x").status_code == 400 and put("empty.md", "  ").status_code == 400
    assert put("bin.txt", "a\x00b").status_code == 400
    assert put("../../evil.md", "# evil text that is long enough to be a paragraph").json()["saved"] == "evil.md"   # path stripped
    assert sorted(os.listdir(os.path.join(rag.USER_CORPORA_DIR, "cars", "docs"))) == ["evil.md", "notes.md"]
    assert client.put("/api/corpora/kestrel/documents", json={"filename": "a.md", "content": "x"}).status_code == 400  # provided corpus
    assert client.delete("/api/corpora/kestrel").status_code == 400
    assert client.get("/api/corpora/..%2Fetc/status").status_code in (400, 404)
    assert client.delete("/api/corpora/cars/documents/evil.md").status_code == 200
    names = {c["name"]: c for c in client.get("/api/corpora").json()}
    assert names["cars"]["user_corpus"] and not names["kestrel"]["user_corpus"]
    assert client.delete("/api/corpora/cars").status_code == 200 and "cars" not in [c["name"] for c in client.get("/api/corpora").json()]


def test_indexing_run_builds_everything_and_is_traced(fake):
    make_cars()
    st = client.get("/api/corpora/cars/status").json()
    assert not st["dense_ready"] and st["graph"]["nodes"] == 0 and st["graph"]["profile"] == "general"
    job = wait_job(client.post("/api/corpora/cars/index", json={"graph": True}).json()["job_id"])
    assert job["status"] == "done", job
    st = client.get("/api/corpora/cars/status").json()
    assert st["dense_ready"] and st["bm25_ready"] and st["graph"]["nodes"] >= 3 and st["graph"]["chunks_done"] == st["chunks"]
    t = client.get("/api/traces/" + job["trace_id"]).json()
    names = [s["name"] for s in t["spans"]]
    assert t["kind"] == "index" and t["outcome"] == "indexed"
    for step in ("load+chunk", "dense.index", "bm25.build", "graph.extract", "graph.chunk", "llm.call"):
        assert step in names
    assert t["totals"]["llm_calls"] == st["chunks"] and t["totals"]["tokens_in"] > 0               # the graph bill, in the trace
    assert "Concept, Person" in fake.requests[0]["prompt"]                                          # general schema for a non-Kestrel corpus
    ans = client.post("/api/ask", json={"corpus": "cars", "question": "What replaced the Aventador?", "retrievers": ["dense", "bm25", "graph"]})
    assert ans.status_code == 200 and any("graph" in c["found_by"] for c in ans.json()["chunks"])  # usable as a retriever


def test_adding_a_document_only_pays_for_the_new_passages(fake):
    make_cars()
    wait_job(client.post("/api/corpora/cars/index", json={"graph": True}).json()["job_id"])
    before = sum("Extract a knowledge graph" in r["prompt"] for r in fake.requests)
    client.put("/api/corpora/cars/documents", json={"filename": "more.md", "content": "The Zeta 3 replaced the Zeta 2 in 2021 on the new platform."})
    job = wait_job(client.post("/api/corpora/cars/index", json={"graph": True}).json()["job_id"])
    after = sum("Extract a knowledge graph" in r["prompt"] for r in fake.requests)
    assert after - before == 1                                                                       # the 1 new passage, not all of them
    t = client.get("/api/traces/" + job["trace_id"]).json()
    assert sum(s["attrs"].get("cache_hit") is True for s in t["spans"] if s["name"] == "graph.chunk") >= 1


def test_a_failing_chunk_is_in_the_trace_and_the_run_is_partial(monkeypatch, fake):
    make_cars()
    real = graph.extract_chunk
    monkeypatch.setattr(graph, "extract_chunk", lambda c, profile=graph.SUPPORT: (_ for _ in ()).throw(llm.LLMError("429 quota")))
    job = wait_job(client.post("/api/corpora/cars/index", json={"graph": True}).json()["job_id"])
    assert job["status"] == "partial"
    t = client.get("/api/traces/" + job["trace_id"]).json()
    bad = [s for s in t["spans"] if s["name"] == "graph.chunk"]
    assert bad and all(s["status"] == "error" and "429" in s["attrs"]["error"] for s in bad) and t["outcome"] == "partial"


def test_index_of_unknown_corpus_is_a_clear_error():
    assert client.post("/api/corpora/nope/index", json={}).status_code == 400


def test_general_profile_validation():
    ok = lambda s, st, r, o, ot: graph._valid(graph.Triple(subject=s, subject_type=st, relation=r, object=o, object_type=ot, evidence="x"), graph.GENERAL)
    assert ok("Revuelto", "Product", "REPLACES", "Aventador", "Product")
    assert not ok("Revuelto", "Product", "REPLACES", "Italy", "Place")      # REPLACES stays same-type
    assert not ok("a", "Service", "PART_OF", "b", "Concept")                # a support type is not allowed here
    assert graph.profile_for("kestrel") is graph.SUPPORT and graph.profile_for("cars") is graph.GENERAL


# ------------------------------------------------------------------------------------------------ concept map explorer

def test_map_and_concept_endpoints(fake):
    make_cars()
    wait_job(client.post("/api/corpora/cars/index", json={"graph": True}).json()["job_id"])
    m = client.get("/api/graph/map", params={"corpus": "cars"}).json()
    assert m["available"] and m["map"]["shown_nodes"] >= 3 and m["map"]["edges"] and m["stats"]["chunks_done"] == m["stats"]["chunks_total"]
    node = next(n for n in m["map"]["nodes"] if n["name"] == "Aventador")
    d = client.get("/api/graph/node", params={"corpus": "cars", "node_id": node["id"]}).json()
    assert {l["relation"] for l in d["links"]} >= {"REPLACES"}
    link = next(l for l in d["links"] if l["relation"] == "REPLACES")
    assert link["direction"] == "in" and link["other"]["name"] == "Revuelto" and link["evidence"][0]["quote"] and link["evidence"][0]["chunk_id"]
    assert d["documents"] == ["notes.md"] and "Aventador" in d["chunks"][0]["text"]
    assert client.get("/api/graph/node", params={"corpus": "cars", "node_id": "Product|nothing"}).status_code == 404
    kinds = [x["kind"] for x in client.get("/api/traces").json()]
    assert "explore" in kinds and kinds.count("index") == 1


def test_map_without_a_graph_explains_what_to_do():
    make_cars()
    m = client.get("/api/graph/map", params={"corpus": "cars"}).json()
    assert m["available"] is False and "My documents" in m["message"]


def test_explorer_limits_the_number_of_nodes():
    ex = {f"d::{i}": [{"subject": f"s{i}", "subject_type": "Service", "relation": "OWNED_BY", "object": "T", "object_type": "Team",
                       "evidence": "q", "subject_aliases": [], "object_aliases": []}] for i in range(30)}
    g = graph.build_graph(ex)
    view = graph.explore(g, limit=10)
    assert view["shown_nodes"] == 10 and view["total_nodes"] == 31 and view["nodes"][0]["name"] == "T"   # best connected first
    assert all(e["src"] in {n["id"] for n in view["nodes"]} for e in view["edges"])


def test_config_endpoint_never_shows_the_key(fake):
    cfg = client.get("/api/config").json()
    assert cfg["model"] == "my-model" and "test-key" not in json.dumps(cfg) and cfg["personal_data_hidden"] is True
