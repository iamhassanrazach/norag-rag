"""RAG core: read a corpus, cut it into chunks, index it in ChromaDB, search it."""
import glob
import hashlib
import os
import zlib

import chromadb
from chromadb.api.types import EmbeddingFunction

import words

CORPORA_DIR = os.getenv("CORPORA_DIR", "corpora")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
MIN_PARAGRAPH_LENGTH = 20
MAX_K = 10

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
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= MIN_PARAGRAPH_LENGTH]
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
    name = f"{corpus}_{chunk_size}_{overlap_percent}_{digest}"
    col = _chroma().get_or_create_collection(name=name, embedding_function=_embedding_function())
    if col.count() == 0:
        chunks = build_chunks(corpus, chunk_size, overlap_percent)
        for start in range(0, len(chunks), 5000):  # ChromaDB accepts ~5000 items per call
            b = chunks[start:start + 5000]
            col.add(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                    metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in b])
    return col


def search(corpus, question, k=5, chunk_size=800, overlap_percent=10):
    if not question.strip():
        raise CorpusError("The question is empty.")
    col = _collection(corpus, chunk_size, overlap_percent)
    k = max(1, min(k, MAX_K, col.count()))
    r = col.query(query_texts=[question], n_results=k)
    return [{"rank": rank + 1, "id": r["ids"][0][rank], "source": r["metadatas"][0][rank]["source"],
             "chunk_index": r["metadatas"][0][rank]["chunk_index"], "distance": r["distances"][0][rank],
             "text": r["documents"][0][rank]} for rank in range(len(r["ids"][0]))]


def load_questions(corpus):
    import json
    path = os.path.join(CORPORA_DIR, corpus, "questions.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
