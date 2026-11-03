import os
import tempfile

os.environ["FAKE_EMBED"] = "1"                       # no model download in tests
os.environ["CHROMA_PATH"] = tempfile.mkdtemp()

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
