import os
import tempfile

os.environ["FAKE_EMBED"] = "1"                       # no model download in tests
os.environ["CHROMA_PATH"] = tempfile.mkdtemp()
os.environ["BM25_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

import evaluate  # noqa: E402
import rag  # noqa: E402
import words  # noqa: E402
from app import app  # noqa: E402

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
    assert res["meta"]["n_questions"] == 10 and res["meta"]["n_scored"] == 8
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
    assert client.post("/api/search", json={"corpus": "own", "question": "x"}).status_code == 400  # empty corpus
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
