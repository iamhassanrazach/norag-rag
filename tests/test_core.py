import os
import tempfile

os.environ["FAKE_EMBED"] = "1"                       # no model download in tests
os.environ["CHROMA_PATH"] = tempfile.mkdtemp()
os.environ["BM25_DIR"] = tempfile.mkdtemp()
os.environ["GRAPH_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

import evaluate  # noqa: E402
import graph  # noqa: E402
import llm  # noqa: E402
import rag  # noqa: E402
import words  # noqa: E402
from app import app  # noqa: E402
import bm25  # noqa: E402

# Belt and braces: whatever was imported first, the tests must NEVER touch the real indexes or graph.
rag.CHROMA_PATH, rag._client = os.environ["CHROMA_PATH"], None
bm25.BM25_DIR = os.environ["BM25_DIR"]
graph.GRAPH_DIR = os.environ["GRAPH_DIR"]

client = TestClient(app)


def test_chunking_overlap_and_end():
    text = ("a" * 100 + "\n\n") * 10
    chunks = rag.chunk_text(text, 300, 10)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[1][:30] == chunks[0][-30:]          # 10 % overlap
    assert len(chunks[-1]) > 30                       # no tiny duplicate tail


def test_tokenize_keeps_identifiers():
    assert "kst-2044" in words.tokenize("Error KST-2044 means")


def test_relevance_needs_source_and_quote():
    ev = [{"source": "a.md", "quote": "hello   world"}]
    assert evaluate.matches({"source": "a.md", "text": "say Hello world!"}, ev) == [0]
    assert evaluate.matches({"source": "b.md", "text": "say hello world"}, ev) == []


def test_kestrel_eval_runs_and_unanswerable_are_excluded():
    res = evaluate.run_eval("kestrel")
    assert res["meta"]["n_questions"] == 12 and res["meta"]["n_scored"] == 10
    assert res["summary"]["overall"]["10"]["hit_rate"] >= res["summary"]["overall"]["1"]["hit_rate"]
    assert {"exact_identifier", "split_answer", "vocabulary", "versions", "two_hop"} <= set(res["summary"]["by_type"])


def test_words_absent_from_kestrel():
    chunks = rag.build_chunks("kestrel", 800, 10)
    rows = {r["word"]: r for r in words.question_words(chunks, "I lost my phone and I am locked out")}
    assert rows["phone"]["exists"] is False and rows["locked"]["exists"] is False


def test_api_search_words_and_errors():
    r = client.post("/api/search", json={"corpus": "kestrel", "question": "KST-2044", "k": 3})
    assert r.status_code == 200 and len(r.json()["chunks"]) == 3
    assert client.get("/api/words/vocabulary", params={"corpus": "kestrel"}).json()["distinct_words"] > 100
    assert client.post("/api/search", json={"corpus": "nope", "question": "x"}).status_code == 400
    assert client.post("/api/search", json={"corpus": "kestrel", "question": "x", "k": 99}).status_code == 422


# ---------------- V3: BM25, RRF, retriever switches ----------------
import fusion  # noqa: E402


def test_bm25_finds_exact_identifier_first():
    # Dense (fake or real) can confuse KST codes; BM25 matches the exact word.
    hits = rag.search("kestrel", "What does error KST-2044 mean?", k=3, retrievers=["bm25"])
    assert "KST-2044" in hits[0]["text"] and hits[0]["found_by"] == {"bm25": 1}


def test_bm25_unknown_words_give_nothing():
    assert rag.search("kestrel", "zzzxqj", k=3, retrievers=["bm25"]) == []


def test_rrf_toy_example():
    a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]          # retriever A: x, y, z
    b = [{"id": "z"}, {"id": "x"}]                        # retriever B: z, x
    out = fusion.rrf({"A": a, "B": b}, k=60, top=3)
    assert [c["id"] for c in out] == ["x", "z", "y"]      # x: 1/61+1/62, z: 1/63+1/61, y: 1/62
    assert out[0]["found_by"] == {"A": 1, "B": 2}
    w = fusion.rrf({"A": a, "B": b}, k=60, weights={"B": 5}, top=1)
    assert w[0]["id"] == "z"                              # a heavy weight on B puts B's first choice on top


def test_dense_only_order_is_unchanged_by_fusion():
    plain = rag.search("kestrel", "refund annual plan", k=5, retrievers=["dense"])
    assert [c["rank"] for c in plain] == [1, 2, 3, 4, 5]
    assert all(set(c["found_by"]) == {"dense"} for c in plain)


def test_eval_with_bm25_and_hybrid_runs_and_reports_cost():
    for r in (["bm25"], ["dense", "bm25"]):
        res = evaluate.run_eval("kestrel", retrievers=r)
        assert res["meta"]["retrievers"] == r and res["summary"]["cost_per_question"]["retrieval_ms"] >= 0
    assert res["summary"]["overall"]["10"]["hit_rate"] >= res["summary"]["overall"]["1"]["hit_rate"]


def test_api_retrievers_validation():
    ok = client.post("/api/search", json={"corpus": "kestrel", "question": "KST-2044", "retrievers": ["dense", "bm25"]})
    assert ok.status_code == 200 and "found_by" in ok.json()["chunks"][0]
    assert client.post("/api/search", json={"corpus": "kestrel", "question": "x", "retrievers": []}).status_code == 400
    assert client.post("/api/search", json={"corpus": "kestrel", "question": "x", "retrievers": ["magic"]}).status_code == 400


def test_empty_corpus_gives_controlled_error(monkeypatch, tmp_path):
    (tmp_path / "empty" / "docs").mkdir(parents=True)       # a corpus folder with no document in it
    monkeypatch.setattr(rag, "CORPORA_DIR", str(tmp_path))
    assert client.post("/api/search", json={"corpus": "empty", "question": "x"}).status_code == 400


# ---------------------------------------------------------------- V3: concept graph (fake LLM, no network)

def T(s, st, r, o, ot, ev="x", sa=(), oa=()):
    return {"subject": s, "subject_type": st, "subject_aliases": list(sa), "relation": r,
            "object": o, "object_type": ot, "object_aliases": list(oa), "evidence": ev}


EXTRACTIONS = {
    "runbook.md::0": [T("KST-2044", "ErrorCode", "RAISED_BY", "query-engine", "Service")],
    "catalog.md::0": [T("query-engine", "Service", "OWNED_BY", "Analytics Engine team", "Team")],
    "catalog.md::1": [T("postroom", "Service", "REPLACES", "mailman", "Service"),
                      T("The postroom service", "Service", "OWNED_BY", "Messaging", "Team")],
    "other.md::0": [T("Billing", "Policy", "REPLACES", "Old billing", "Policy")],
}


def test_resolution_merges_same_type_same_name_and_keeps_provenance():
    g = graph.build_graph(EXTRACTIONS)
    names = sorted(n["name"] for n in g["nodes"].values() if n["type"] == "Service")
    assert names == ["mailman", "postroom", "query-engine"]       # "The postroom service" merged with "postroom"
    qe = next(n for n in g["nodes"].values() if n["name"] == "query-engine")
    assert qe["chunks"] == {"runbook.md::0", "catalog.md::0"}      # provenance of the node


def test_alias_merges_nodes():
    g = graph.build_graph({"a::0": [T("mail", "Service", "REPLACES", "old", "Service", oa=["mailman"])],
                           "a::1": [T("x", "Service", "DEPENDS_ON", "mailman", "Service")]})
    assert len([n for n in g["nodes"].values() if n["type"] == "Service"]) == 3   # mail, old+mailman merged, x


def test_two_hop_walk_reaches_the_owner_chunk():
    g = graph.build_graph(EXTRACTIONS)
    chunks = {cid: {"id": cid, "source": cid.split("::")[0], "chunk_index": 0, "text": cid} for cid in EXTRACTIONS}
    got, seeds = graph.retrieve(g, chunks, "Which team owns the service behind error kst 2044?")
    ids = [c["id"] for c in got]
    assert seeds == ["KST-2044"]                                   # "kst 2044" matched through the squashed form
    assert ids[:2] == ["runbook.md::0", "catalog.md::0"]           # hop 0 chunk first, then the hop-2 owner chunk
    assert any("OWNED_BY" in p for p in got[1]["path"])
    assert "other.md::0" not in ids


def test_extract_chunk_drops_bad_triples_and_retries_bad_json(monkeypatch):
    answers = iter(["not json",
                    '{"triples": [{"subject": "a", "subject_type": "Service", "relation": "OWNED_BY", "object": "b",'
                    ' "object_type": "Team", "evidence": "q"}, {"subject": "a", "subject_type": "Planet",'
                    ' "relation": "OWNED_BY", "object": "b", "object_type": "Team", "evidence": "q"}]}'])
    monkeypatch.setattr(llm, "_complete", lambda prompt, **kw: next(answers))
    kept, dropped = graph.extract_chunk({"id": "x::0", "source": "x", "text": "t"})
    assert len(kept) == 1 and dropped == 1


def test_build_cache_survives_a_failure_and_resumes(monkeypatch):
    chunks = [{"id": "d::0", "source": "d", "text": "a"}, {"id": "d::1", "source": "d", "text": "b"}]

    def fake(chunk, profile=None):
        if chunk["id"] == "d::1":
            raise llm.LLMError("429")
        return [T("s", "Service", "OWNED_BY", "t", "Team")], 0
    monkeypatch.setattr(graph, "extract_chunk", fake)
    c = graph.build_cache("unit-" + str(id(chunks)), chunks, log=None)
    assert "d::0" in c["extractions"] and "d::1" in c["failed"]
    monkeypatch.setattr(graph, "extract_chunk", lambda ch, profile=None: ([], 0))
    c = graph.build_cache("unit-" + str(id(chunks)), chunks, log=None)                 # only the failed chunk is called again
    assert "d::1" in c["extractions"] and not c["failed"]


def test_graph_retriever_without_a_graph_gives_a_controlled_error():
    import pytest
    with pytest.raises(rag.CorpusError, match="python graph.py build"):
        rag.search("kestrel", "who owns query-engine?", retrievers=("graph",))


def test_search_fuses_graph_with_dense(monkeypatch):
    name = rag._index_name("kestrel", 800, 20)
    chunks = rag.build_chunks("kestrel", 800, 20)
    target = next(c for c in chunks if c["source"] == "service-catalog.md" and "query" in c["text"].lower())
    graph.save_cache(name, {"extractions": {target["id"]: [T("query-engine", "Service", "OWNED_BY", "Analytics Engine", "Team")]},
                            "failed": {}, "cost": {"calls": 1, "tokens": 1, "seconds": 0.1, "dropped_triples": 0}})
    out = rag.search("kestrel", "who owns query-engine?", k=5, chunk_size=800, overlap_percent=20,
                     retrievers=("dense", "graph"), weights={"graph": 2})
    hit = next(c for c in out if c["id"] == target["id"])
    assert "graph" in hit["found_by"] and hit["path"]
    assert out[0]["id"] == target["id"]                               # weight 2 puts the graph's chunk first


def test_wrong_direction_triples_are_dropped(monkeypatch):
    bad = ('{"triples": [{"subject": "Platform", "subject_type": "Team", "relation": "OWNED_BY", "object": "gateway",'
           ' "object_type": "Service", "evidence": "q"}, {"subject": "gateway", "subject_type": "Service",'
           ' "relation": "owned_by", "object": "Platform", "object_type": "team", "evidence": "q"}]}')
    monkeypatch.setattr(llm, "_complete", lambda prompt, **kw: bad)
    kept, dropped = graph.extract_chunk({"id": "x::0", "source": "x", "text": "t"})
    assert [(t["subject"], t["object"]) for t in kept] == [("gateway", "Platform")] and dropped == 1


def test_headings_are_kept_in_chunks():
    text = "### query-engine\n\nRuns dataset queries and enforces the time limit.\n\nshort"
    assert "### query-engine" in rag.chunk_text(text, 800, 0)[0]
    assert "short" not in rag.chunk_text(text, 800, 0)[0]
