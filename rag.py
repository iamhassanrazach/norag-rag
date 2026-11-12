"""RAG core: read a corpus, cut it into chunks, index it in ChromaDB, search it."""
import glob
import hashlib
import os
import zlib

import chromadb
from chromadb.api.types import EmbeddingFunction

import bm25
import fusion
import graph
import words

CORPORA_DIR = os.getenv("CORPORA_DIR", "corpora")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
MIN_PARAGRAPH_LENGTH = 20
# The base code drops every paragraph shorter than 20 characters. In Markdown that also drops the HEADINGS
# ("### query-engine"), so the text no longer says which service a paragraph is about. We keep them by default
# (KEEP_HEADINGS=0 gives the old behaviour, for a before/after comparison).
KEEP_HEADINGS = os.getenv("KEEP_HEADINGS", "1") != "0"
MAX_K = 10
MAX_CANDIDATES = 50
RETRIEVERS = ("dense", "bm25", "graph")

_client = None
_ef = None


class CorpusError(Exception):
    """Something the user can understand and fix (unknown corpus, no documents...)."""


class HashEmbedding(EmbeddingFunction):
    """TESTS ONLY (FAKE_EMBED=1): counts words in 256 buckets. No model download, no quality."""
    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 256
            for w in words.tokenize(text):
                v[zlib.crc32(w.encode()) % 256] += 1
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    @staticmethod
    def name():
        return "hash-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


def _embedding_function():
    global _ef
    if _ef is None:
        if os.getenv("FAKE_EMBED"):
            _ef = HashEmbedding()
        else:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            _ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return _ef


def _chroma():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def list_corpora():
    if not os.path.isdir(CORPORA_DIR):
        return []
    return sorted(d for d in os.listdir(CORPORA_DIR) if os.path.isdir(os.path.join(CORPORA_DIR, d, "docs")))


def read_documents(corpus):
    docs_dir = os.path.join(CORPORA_DIR, corpus, "docs")
    if corpus not in list_corpora():
        raise CorpusError(f"Unknown corpus '{corpus}'. Available: {list_corpora()}")
    files = sorted(f for f in glob.glob(os.path.join(docs_dir, "*")) if f.endswith((".md", ".txt")))
    if not files:
        raise CorpusError(f"Corpus '{corpus}' has no documents yet (put .md or .txt files in {docs_dir}).")
    docs = {}
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            docs[os.path.basename(f)] = fh.read()
    return docs


def chunk_text(text, chunk_size, overlap_percent):
    """Same idea as ask_rag_google.py: a window of chunk_size characters slides over the text,
    each chunk starts with the last overlap_percent % of the previous one."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= MIN_PARAGRAPH_LENGTH
                  or (KEEP_HEADINGS and p.strip().startswith("#"))]
    text = "\n\n".join(paragraphs)
    step = max(1, chunk_size - chunk_size * overlap_percent // 100)
    chunks = []
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):  # this chunk already reaches the end: stop (no tiny duplicate tail)
            break
    return chunks


def build_chunks(corpus, chunk_size, overlap_percent):
    """All chunks of the corpus (with their source file), computed from the files."""
    out = []
    for source, text in read_documents(corpus).items():
        for i, t in enumerate(chunk_text(text, chunk_size, overlap_percent)):
            out.append({"id": f"{source}::{i}", "source": source, "chunk_index": i, "text": t})
    return out


def _collection(corpus, chunk_size, overlap_percent):
    """One collection per corpus + chunking settings + document CONTENT (hash).
    Edit a document and a fresh index is built; no stale index."""
    docs = read_documents(corpus)
    digest = hashlib.md5("".join(f"{k}{v}" for k, v in docs.items()).encode()).hexdigest()[:8]
    name = f"{corpus}_{chunk_size}_{overlap_percent}_{digest}{'_h' if KEEP_HEADINGS else ''}" + ("_fake" if os.getenv("FAKE_EMBED") else "")
    col = _chroma().get_or_create_collection(name=name, embedding_function=_embedding_function())
    if col.count() == 0:
        chunks = build_chunks(corpus, chunk_size, overlap_percent)
        for start in range(0, len(chunks), 5000):  # ChromaDB accepts ~5000 items per call
            b = chunks[start:start + 5000]
            col.add(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                    metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in b])
    return col


def _index_name(corpus, chunk_size, overlap_percent):
    docs = read_documents(corpus)
    digest = hashlib.md5("".join(f"{k}{v}" for k, v in docs.items()).encode()).hexdigest()[:8]
    return f"{corpus}_{chunk_size}_{overlap_percent}_{digest}{'_h' if KEEP_HEADINGS else ''}"


def _dense(corpus, question, n, chunk_size, overlap_percent):
    col = _collection(corpus, chunk_size, overlap_percent)
    n = max(1, min(n, col.count()))
    r = col.query(query_texts=[question], n_results=n)
    return [{"rank": rank + 1, "id": r["ids"][0][rank], "source": r["metadatas"][0][rank]["source"],
             "chunk_index": r["metadatas"][0][rank]["chunk_index"], "distance": r["distances"][0][rank],
             "text": r["documents"][0][rank]} for rank in range(len(r["ids"][0]))]


def _bm25(corpus, question, n, chunk_size, overlap_percent):
    name = _index_name(corpus, chunk_size, overlap_percent)
    chunks = build_chunks(corpus, chunk_size, overlap_percent)
    by_id = {c["id"]: c for c in chunks}
    index = bm25.get_index(name, chunks)
    return [{**by_id[cid], "rank": rank, "bm25_score": score}
            for rank, (cid, score) in enumerate(index.search(question, n), start=1)]


def _graph(corpus, question, n, chunk_size, overlap_percent):
    """Graph retriever: uses the graph that `python graph.py build <corpus>` saved. No LLM call here."""
    name = _index_name(corpus, chunk_size, overlap_percent)
    chunks = build_chunks(corpus, chunk_size, overlap_percent)
    g = graph.get_graph(name, chunks)
    if not g["nodes"]:
        raise CorpusError(f"No concept graph for '{corpus}' with these chunk settings. "
                          f"Build it first: python graph.py build {corpus} --chunk-size {chunk_size} --overlap {overlap_percent}")
    got, _seeds = graph.retrieve(g, {c["id"]: c for c in chunks}, question, n)
    return got


def prepare(corpus, chunk_size=800, overlap_percent=10, retrievers=("dense",)):
    """Build the indexes now (so that timings do not include the one-off build)."""
    if "dense" in retrievers:
        _collection(corpus, chunk_size, overlap_percent)
    if "bm25" in retrievers:
        _bm25(corpus, "warm up", 1, chunk_size, overlap_percent)
    if "graph" in retrievers:
        _graph(corpus, "warm up", 1, chunk_size, overlap_percent)


def search(corpus, question, k=5, chunk_size=800, overlap_percent=10,
           retrievers=("dense",), candidates=20, rrf_k=60, weights=None):
    """Ask every retriever that is switched on for `candidates` chunks, fuse with RRF, keep the top k.
    Each returned chunk has 'found_by': {retriever: rank}. With dense only, the order is exactly V2's."""
    if not question.strip():
        raise CorpusError("The question is empty.")
    retrievers = tuple(retrievers)
    if not retrievers or any(r not in RETRIEVERS for r in retrievers):
        raise CorpusError(f"Retrievers must be a non-empty subset of {list(RETRIEVERS)}.")
    k = max(1, min(k, MAX_K))
    candidates = max(k, min(candidates, MAX_CANDIDATES))
    lists = {}
    if "dense" in retrievers:
        lists["dense"] = _dense(corpus, question, candidates, chunk_size, overlap_percent)
    if "bm25" in retrievers:
        lists["bm25"] = _bm25(corpus, question, candidates, chunk_size, overlap_percent)
    if "graph" in retrievers:
        lists["graph"] = _graph(corpus, question, candidates, chunk_size, overlap_percent)
    return fusion.rrf(lists, k=rrf_k, weights=weights, top=k)


def load_questions(corpus):
    import json
    path = os.path.join(CORPORA_DIR, corpus, "questions.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
