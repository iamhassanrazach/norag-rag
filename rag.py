"""RAG core: read a corpus, cut it into chunks, index it in ChromaDB, search it."""
import glob
import hashlib
import os
import re
import shutil
import zlib
from collections import Counter

import chromadb
from chromadb.api.types import EmbeddingFunction

import bm25
import fusion
import graph
import tracing
import words

CORPORA_DIR = os.getenv("CORPORA_DIR", "corpora")                  # the corpora that come with the project (read-only for the app)
USER_CORPORA_DIR = os.getenv("USER_CORPORA_DIR", "user_corpora")   # the corpora created from the UI (documents uploaded by the user)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_PARAGRAPH_LENGTH = 20
# The base code drops every paragraph shorter than 20 characters. In Markdown that also drops the HEADINGS
# ("### query-engine"), so the text no longer says which service a paragraph is about. We keep them by default
# (KEEP_HEADINGS=0 gives the old behaviour, for a before/after comparison).
KEEP_HEADINGS = os.getenv("KEEP_HEADINGS", "1") != "0"
MAX_K = 10
MAX_CANDIDATES = 50
RETRIEVERS = ("dense", "bm25", "graph")
MAX_DOC_BYTES = 2_000_000
MAX_DOCS = 100
_LOOKUP_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,39}$")   # no dot, no slash: a corpus name can never be a path
_NAME_RX = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,39}$")

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
            _ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _ef


def _chroma():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


# ----------------------------------------------------------------------------- corpora (provided + uploaded)

def corpus_dir(name):
    """Folder of a corpus, or None. Looks in corpora/ then in user_corpora/."""
    if not isinstance(name, str) or not _LOOKUP_RX.match(name):
        return None
    for base in (CORPORA_DIR, USER_CORPORA_DIR):
        path = os.path.join(base, name)
        if os.path.isdir(os.path.join(path, "docs")):
            return path
    return None


def is_user_corpus(name):
    path = corpus_dir(name)
    return bool(path) and os.path.abspath(os.path.dirname(path)) == os.path.abspath(USER_CORPORA_DIR)


def list_corpora():
    found = set()
    for base in (CORPORA_DIR, USER_CORPORA_DIR):
        if os.path.isdir(base):
            found |= {d for d in os.listdir(base) if _LOOKUP_RX.match(d) and os.path.isdir(os.path.join(base, d, "docs"))}
    return sorted(found)


def create_corpus(name):
    name = (name or "").strip().lower()
    if not _NAME_RX.match(name):
        raise CorpusError("A name has 2 to 40 characters: letters, digits, - or _ (no spaces).")
    if corpus_dir(name):
        raise CorpusError(f"A corpus called '{name}' already exists.")
    os.makedirs(os.path.join(USER_CORPORA_DIR, name, "docs"))
    return name


def _user_docs_dir(corpus):
    if not is_user_corpus(corpus):
        raise CorpusError(f"'{corpus}' is a provided corpus: it cannot be changed from the app. Create your own corpus.")
    return os.path.join(corpus_dir(corpus), "docs")


def add_document(corpus, filename, content):
    """Save one uploaded document (text) into one of the user's corpora."""
    docs_dir = _user_docs_dir(corpus)
    safe = re.sub(r"[^\w.\- ()]", "_", os.path.basename((filename or "").replace("\\", "/"))).strip(" .")
    if not safe.lower().endswith((".md", ".txt")):
        raise CorpusError("Only .md and .txt documents are supported.")
    if not content.strip() or "\x00" in content:
        raise CorpusError(f"'{safe}' is empty or is not a text file.")
    if len(content.encode("utf-8")) > MAX_DOC_BYTES:
        raise CorpusError(f"'{safe}' is too big (limit {MAX_DOC_BYTES // 1_000_000} MB).")
    existing = [f for f in os.listdir(docs_dir)]
    if safe not in existing and len(existing) >= MAX_DOCS:
        raise CorpusError(f"A corpus holds at most {MAX_DOCS} documents.")
    with open(os.path.join(docs_dir, safe), "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return safe


def remove_document(corpus, filename):
    path = os.path.join(_user_docs_dir(corpus), os.path.basename(filename))
    if not os.path.isfile(path):
        raise CorpusError(f"No document '{filename}' in '{corpus}'.")
    os.remove(path)


def delete_corpus(corpus):
    _user_docs_dir(corpus)  # refuses provided corpora
    shutil.rmtree(corpus_dir(corpus))


def read_documents(corpus):
    path = corpus_dir(corpus)
    if not path:
        raise CorpusError(f"Unknown corpus '{corpus}'. Available: {list_corpora()}")
    docs_dir = os.path.join(path, "docs")
    files = sorted(f for f in glob.glob(os.path.join(docs_dir, "*")) if f.endswith((".md", ".txt")))
    if not files:
        raise CorpusError(f"Corpus '{corpus}' has no documents yet (add .md or .txt documents first).")
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


def get_chunk(corpus, chunk_size, overlap_percent, chunk_id):
    chunk = next((c for c in build_chunks(corpus, chunk_size, overlap_percent) if c["id"] == chunk_id), None)
    if not chunk:
        raise CorpusError(f"No chunk '{chunk_id}' in '{corpus}' with these chunk settings.")
    return chunk


# ----------------------------------------------------------------------------- indexes

def corpus_version(corpus):
    """Eight characters that change whenever a document of the corpus changes."""
    docs = read_documents(corpus)
    return hashlib.md5("".join(f"{k}{v}" for k, v in docs.items()).encode()).hexdigest()[:8]


def _index_name(corpus, chunk_size, overlap_percent):
    return f"{corpus}_{chunk_size}_{overlap_percent}_{corpus_version(corpus)}{'_h' if KEEP_HEADINGS else ''}"


def _collection_name(corpus, chunk_size, overlap_percent):
    """One collection per corpus + chunking settings + document CONTENT (hash).
    Edit a document and a fresh index is built; no stale index."""
    return _index_name(corpus, chunk_size, overlap_percent) + ("_fake" if os.getenv("FAKE_EMBED") else "")


def _collection(corpus, chunk_size, overlap_percent):
    col = _chroma().get_or_create_collection(name=_collection_name(corpus, chunk_size, overlap_percent),
                                             embedding_function=_embedding_function())
    if col.count() == 0:
        chunks = build_chunks(corpus, chunk_size, overlap_percent)
        with tracing.span("dense.embed", vectors=len(chunks), embedding_model=EMBED_MODEL):
            for start in range(0, len(chunks), 5000):  # ChromaDB accepts ~5000 items per call
                b = chunks[start:start + 5000]
                col.add(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in b])
    return col


def _dense_ready(corpus, chunk_size, overlap_percent):
    name = _collection_name(corpus, chunk_size, overlap_percent)
    if name not in {getattr(c, "name", c) for c in _chroma().list_collections()}:
        return False
    return _chroma().get_collection(name=name, embedding_function=_embedding_function()).count() > 0


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
    """Graph retriever: uses the graph that was built for this corpus (UI: My documents, or python graph.py build)."""
    name = _index_name(corpus, chunk_size, overlap_percent)
    chunks = build_chunks(corpus, chunk_size, overlap_percent)
    g = graph.get_graph(name, chunks, profile=graph.profile_for(corpus))
    if not g["nodes"]:
        raise CorpusError(f"There is no concept graph for '{corpus}' yet. Build it in the 'My documents' tab "
                          f"(or: python graph.py build {corpus} --chunk-size {chunk_size} --overlap {overlap_percent}).")
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


def index_status(corpus, chunk_size=800, overlap_percent=20):
    """What is ready for this corpus and these chunk settings (used by the 'My documents' tab)."""
    docs = read_documents(corpus)
    chunks = build_chunks(corpus, chunk_size, overlap_percent)
    name = _index_name(corpus, chunk_size, overlap_percent)
    g = graph.get_graph(name, chunks, profile=graph.profile_for(corpus))
    return {"corpus": corpus, "user_corpus": is_user_corpus(corpus), "documents": sorted(docs), "chunks": len(chunks),
            "dense_ready": _dense_ready(corpus, chunk_size, overlap_percent),
            "bm25_ready": os.path.exists(os.path.join(bm25.BM25_DIR, f"{name}.json")),
            "graph": {"profile": graph.profile_for(corpus)["name"], "chunks_done": g["chunks_done"],
                      "chunks_total": g["chunks_total"], "nodes": len(g["nodes"]), "edges": len(g["edges"]),
                      "failed": len(g["failed"])}}


def index_corpus(corpus, chunk_size=800, overlap_percent=20, with_graph=False, progress=None):
    """One INDEXING RUN: chunks, dense vectors, BM25, and (optionally) the concept graph. One trace for the whole run."""
    progress = progress or (lambda *a, **k: None)
    graph_profile = graph.profile_for(corpus)
    cfg = {"corpus": corpus, "chunk_size": chunk_size, "chunk_overlap": overlap_percent, "embedding_model": EMBED_MODEL,
           "headings_kept": KEEP_HEADINGS, "graph": with_graph, "graph_profile": graph_profile["name"] if with_graph else None}
    with tracing.trace("index", corpus=corpus, config=cfg, fingerprint=tracing.fingerprint(cfg)) as t:
        progress("chunking", trace_id=t.trace_id)
        with tracing.span("load+chunk") as sp:
            docs = read_documents(corpus)
            chunks = build_chunks(corpus, chunk_size, overlap_percent)
            sp.set(documents=len(docs), chunks=len(chunks), chunks_per_document=dict(Counter(c["source"] for c in chunks)))
        progress("embedding")
        with tracing.span("dense.index") as sp:
            sp.set(reused=_dense_ready(corpus, chunk_size, overlap_percent))
            sp.set(vectors=_collection(corpus, chunk_size, overlap_percent).count())
        progress("keyword index")
        with tracing.span("bm25.build") as sp:
            name = _index_name(corpus, chunk_size, overlap_percent)
            sp.set(reused=os.path.exists(os.path.join(bm25.BM25_DIR, f"{name}.json")))
            _bm25(corpus, "warm up", 1, chunk_size, overlap_percent)
        result = {"corpus": corpus, "chunks": len(chunks), "graph": None}
        if with_graph:
            progress("concept graph", 0, len(chunks))
            with tracing.span("graph.extract") as sp:
                prof = graph_profile
                cache = graph.build_cache(name, chunks, log=None, profile=prof,
                                          progress=lambda done, total, cid: progress("concept graph", done, total))
                g = graph.get_graph(name, chunks, profile=prof)
                sp.set(profile=prof["name"], chunks=len(chunks), failed=len(cache["failed"]), nodes=len(g["nodes"]),
                       edges=len(g["edges"]), dropped_triples=cache["cost"]["dropped_triples"])
                result["graph"] = {"nodes": len(g["nodes"]), "edges": len(g["edges"]), "failed": len(cache["failed"]),
                                   "chunks_total": len(chunks)}
            t.set(outcome="partial" if cache["failed"] else "indexed")
        else:
            t.set(outcome="indexed")
        result["trace_id"] = t.trace_id
    return result


# ----------------------------------------------------------------------------- search

def _top(chunks, score_key):
    return [{"id": c["id"], "rank": c["rank"], "score": round(c[score_key], 4) if score_key in c else None} for c in chunks[:10]]


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
    with tracing.span("retrieve", retrievers=list(retrievers), k=k, candidates=candidates) as sp:
        for name, fn, score_key in (("dense", _dense, "distance"), ("bm25", _bm25, "bm25_score"), ("graph", _graph, "graph_score")):
            if name in retrievers:
                with tracing.span(f"retrieve.{name}") as s:
                    lists[name] = fn(corpus, question, candidates, chunk_size, overlap_percent)
                    s.set(n_results=len(lists[name]), top=_top(lists[name], score_key), score_name=score_key)
        with tracing.span("fusion", rrf_k=rrf_k, weights=weights or {}) as s:
            out = fusion.rrf(lists, k=rrf_k, weights=weights, top=k)
            s.set(fused=[{"id": c["id"], "rank": c["rank"], "found_by": c["found_by"], "rrf_score": c["rrf_score"]} for c in out])
        sp.set(returned=[c["id"] for c in out])
    return out


def load_questions(corpus):
    import json
    path = corpus_dir(corpus)
    path = os.path.join(path, "questions.jsonl") if path else ""
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
