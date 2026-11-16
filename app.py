"""Web API + front-end.  Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000"""
import os
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import evaluate
import graph
import llm
import pii
import rag
import tracing
import words

app = FastAPI(title="RAG V4")


class Config(BaseModel):
    corpus: str
    chunk_size: int = Field(800, ge=100, le=4000)
    overlap_percent: int = Field(20, ge=0, le=50)


class SearchIn(Config):
    question: str
    k: int = Field(5, ge=1, le=10)
    retrievers: list[str] = ["dense"]
    candidates: int = Field(20, ge=1, le=50)
    rrf_k: int = Field(60, ge=1, le=1000)
    weights: dict[str, float] = {}


class EvalIn(Config):
    with_answers: bool = False
    answer_k: int = Field(5, ge=1, le=10)
    retrievers: list[str] = ["dense"]
    candidates: int = Field(20, ge=1, le=50)
    rrf_k: int = Field(60, ge=1, le=1000)
    weights: dict[str, float] = {}


class QuestionWordsIn(Config):
    question: str


class NewCorpusIn(BaseModel):
    name: str


class DocumentIn(BaseModel):
    filename: str
    content: str


class IndexIn(BaseModel):
    chunk_size: int = Field(800, ge=100, le=4000)
    overlap_percent: int = Field(20, ge=0, le=50)
    graph: bool = False


@app.exception_handler(rag.CorpusError)
async def corpus_error(_, e):
    return JSONResponse(status_code=400, content={"detail": str(e)})


@app.exception_handler(llm.LLMError)
async def llm_error(_, e):
    return JSONResponse(status_code=502, content={"detail": str(e)})


@app.get("/api/config")
def config():
    """What the app is plugged into (shown in the page footer). Never the key."""
    pin, pout = llm.price()
    return {"model": llm.model(), "server": llm.url(), "personal_data_hidden": pii.enabled(),
            "price_per_million_tokens": {"in": pin, "out": pout}, "trace_retention_days": tracing.RETENTION_DAYS}


# ----------------------------------------------------------------------------- corpora and documents

@app.get("/api/corpora")
def corpora():
    out = []
    for c in rag.list_corpora():
        try:
            n_docs = len(rag.read_documents(c))
        except rag.CorpusError:
            n_docs = 0
        out.append({"name": c, "documents": n_docs, "questions": len(rag.load_questions(c)), "user_corpus": rag.is_user_corpus(c)})
    return out


@app.post("/api/corpora")
def create_corpus(body: NewCorpusIn):
    return {"name": rag.create_corpus(body.name)}


@app.delete("/api/corpora/{name}")
def delete_corpus(name: str):
    rag.delete_corpus(name)
    return {"deleted": name}


@app.get("/api/corpora/{name}/status")
def corpus_status(name: str, chunk_size: int = 800, overlap_percent: int = 20):
    if not rag.corpus_dir(name):
        raise rag.CorpusError(f"Unknown corpus '{name}'.")
    try:
        return rag.index_status(name, chunk_size, overlap_percent)
    except rag.CorpusError:  # a corpus with no document yet
        return {"corpus": name, "user_corpus": rag.is_user_corpus(name), "documents": [], "chunks": 0, "dense_ready": False,
                "bm25_ready": False, "graph": {"profile": graph.profile_for(name)["name"], "chunks_done": 0, "chunks_total": 0,
                                               "nodes": 0, "edges": 0, "failed": 0}}


@app.put("/api/corpora/{name}/documents")
def add_document(name: str, body: DocumentIn):
    return {"saved": rag.add_document(name, body.filename, body.content)}


@app.delete("/api/corpora/{name}/documents/{filename}")
def remove_document(name: str, filename: str):
    rag.remove_document(name, filename)
    return {"deleted": filename}


# ----------------------------------------------------------------------------- indexing jobs (run in the background)

JOBS = {}


def _run_job(job_id, corpus, body):
    job = JOBS[job_id]

    def progress(stage, done=None, total=None, trace_id=None):
        job.update(stage=stage, done=done, total=total)
        if trace_id:
            job["trace_id"] = trace_id

    try:
        res = rag.index_corpus(corpus, body.chunk_size, body.overlap_percent, body.graph, progress)
        job.update(status="partial" if res["graph"] and res["graph"]["failed"] else "done", result=res, trace_id=res["trace_id"])
    except (rag.CorpusError, llm.LLMError) as e:
        job.update(status="error", error=str(e))
    except Exception as e:  # a bug must show up as a message, not as a job that never ends
        job.update(status="error", error=f"Unexpected error: {type(e).__name__}: {e}")


@app.post("/api/corpora/{name}/index")
def start_index(name: str, body: IndexIn):
    rag.read_documents(name)  # unknown corpus / no documents: a clear error now, not in the background
    for jid, j in JOBS.items():
        if j["corpus"] == name and j["status"] == "running":
            return {"job_id": jid}
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"corpus": name, "status": "running", "stage": "starting", "done": None, "total": None,
                    "trace_id": None, "error": None, "result": None}
    threading.Thread(target=_run_job, args=(job_id, name, body), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job.")
    return JOBS[job_id]


# ----------------------------------------------------------------------------- questions

@app.get("/api/questions")
def questions(corpus: str):
    return rag.load_questions(corpus)


def _config_of(body):
    """Everything that controls the behaviour of the pipeline: recorded with the trace; its hash is the fingerprint."""
    cfg = {"prompt_version": llm.ANSWER_PROMPT_VERSION, "llm": llm.model(), "embedding_model": rag.EMBED_MODEL,
           "chunk_size": body.chunk_size, "chunk_overlap": body.overlap_percent, "top_k": body.k,
           "retrievers": list(body.retrievers), "candidates": body.candidates, "rrf_k": body.rrf_k, "weights": body.weights,
           "personal_data_hidden": pii.enabled()}
    try:
        cfg["corpus"] = f"{body.corpus}@{rag.corpus_version(body.corpus)}"
    except rag.CorpusError:
        cfg["corpus"] = body.corpus
    return cfg


def _fail(e, trace_id):
    return JSONResponse(status_code=400 if isinstance(e, rag.CorpusError) else 502,
                        content={"detail": str(e), "trace_id": trace_id})


@app.post("/api/search")
def search(body: SearchIn):
    cfg = _config_of(body)
    try:
        with tracing.trace("search", corpus=body.corpus, question=body.question, config=cfg,
                           fingerprint=tracing.fingerprint(cfg)) as t:
            chunks = rag.search(body.corpus, body.question, body.k, body.chunk_size, body.overlap_percent,
                                body.retrievers, body.candidates, body.rrf_k, body.weights)
            t.set(outcome="found" if chunks else "nothing_found", retrieved=[c["id"] for c in chunks])
    except rag.CorpusError as e:
        return _fail(e, t.trace_id)
    return {"chunks": chunks, "trace_id": t.trace_id}


@app.post("/api/ask")
def ask(body: SearchIn):
    cfg = _config_of(body)
    red = pii.Redactor()
    try:
        with tracing.trace("question", corpus=body.corpus, question=body.question, config=cfg,
                           fingerprint=tracing.fingerprint(cfg)) as t:
            chunks = rag.search(body.corpus, body.question, body.k, body.chunk_size, body.overlap_percent,
                                body.retrievers, body.candidates, body.rrf_k, body.weights)
            t.set(retrieved=[c["id"] for c in chunks])
            try:
                a = llm.answer_with_context(body.question, chunks, red)
            except llm.InvalidAnswer:
                t.set(outcome="invalid_output")
                raise
            t.set(outcome="answered" if a.in_documents else "abstained", answer=a.safe_answer,
                  personal_data_hidden=red.summary())
    except (rag.CorpusError, llm.LLMError) as e:
        return _fail(e, t.trace_id)
    rec = tracing.get_trace(t.trace_id) or {"duration_ms": None, "totals": {}}
    return {"answer": a.answer, "in_documents": a.in_documents, "chunks": chunks, "trace_id": t.trace_id,
            "privacy": red.summary(), "summary": {"duration_ms": rec["duration_ms"], **rec["totals"]}}


# ----------------------------------------------------------------------------- traces and metrics

@app.get("/api/traces")
def traces(limit: int = 50, kind: str = ""):
    return tracing.list_traces(min(max(limit, 1), 200), kind or None)


@app.get("/api/traces/{trace_id}")
def one_trace(trace_id: str):
    t = tracing.get_trace(trace_id)
    if not t:
        raise HTTPException(404, "This trace does not exist (any more): traces are deleted after "
                                 f"{tracing.RETENTION_DAYS} days.")
    return t


@app.get("/api/metrics")
def metrics(days: int = 7):
    return tracing.metrics(min(max(days, 1), 90))


@app.get("/api/chunk")
def chunk(corpus: str, chunk_id: str, chunk_size: int = 800, overlap_percent: int = 20):
    """The text of one chunk (traces keep chunk ids, not text: the text is in the index)."""
    return rag.get_chunk(corpus, chunk_size, overlap_percent, chunk_id)


# ----------------------------------------------------------------------------- evaluation (V2/V3)

@app.post("/api/eval/run")
def run_eval(body: EvalIn):
    result = evaluate.run_eval(body.corpus, body.chunk_size, body.overlap_percent, body.with_answers, body.answer_k,
                               retrievers=body.retrievers, candidates=body.candidates, rrf_k=body.rrf_k, weights=body.weights)
    result["saved_to"] = evaluate.save(result)
    return result


# ----------------------------------------------------------------------------- concept graph and the concept map explorer

def _graph_of(corpus, chunk_size, overlap_percent):
    chunks = rag.build_chunks(corpus, chunk_size, overlap_percent)
    g = graph.get_graph(rag._index_name(corpus, chunk_size, overlap_percent), chunks, profile=graph.profile_for(corpus))
    return g, chunks


@app.get("/api/graph")
def graph_info(corpus: str, chunk_size: int = 800, overlap_percent: int = 20, question: str = ""):
    """Stats of the concept graph (build cost included) and, if a question is given, the seeds it matches."""
    g, chunks = _graph_of(corpus, chunk_size, overlap_percent)
    out = {"stats": graph.stats(g)}
    if question:
        got, seeds = graph.retrieve(g, {c["id"]: c for c in chunks}, question, 10)
        out.update(seeds=seeds, chunks=[{"id": c["id"], "graph_score": c["graph_score"], "path": c["path"]} for c in got])
    return out


@app.get("/api/graph/map")
def graph_map(corpus: str, chunk_size: int = 800, overlap_percent: int = 20):
    with tracing.trace("explore", corpus=corpus, action="map") as t:
        with tracing.span("graph.load") as sp:
            g, _ = _graph_of(corpus, chunk_size, overlap_percent)
            sp.set(nodes=len(g["nodes"]), edges=len(g["edges"]), chunks_done=g["chunks_done"], chunks_total=g["chunks_total"])
        if not g["nodes"]:
            t.set(outcome="no_graph")
            return {"available": False, "chunks_done": g["chunks_done"], "chunks_total": g["chunks_total"],
                    "message": "There is no concept graph for these documents yet. Build it in the 'My documents' tab."}
        with tracing.span("graph.view") as sp:
            view = graph.explore(g)
            sp.set(shown_nodes=view["shown_nodes"], total_nodes=view["total_nodes"])
        t.set(outcome="shown")
    return {"available": True, "map": view, "stats": graph.stats(g), "trace_id": t.trace_id}


@app.get("/api/graph/node")
def graph_node(corpus: str, node_id: str, chunk_size: int = 800, overlap_percent: int = 20):
    with tracing.trace("explore", corpus=corpus, action="concept", concept=node_id) as t:
        with tracing.span("graph.load"):
            g, chunks = _graph_of(corpus, chunk_size, overlap_percent)
        with tracing.span("graph.concept") as sp:
            detail = graph.node_detail(g, node_id, {c["id"]: c for c in chunks})
            if detail is None:
                t.set(outcome="unknown_concept")
                raise HTTPException(404, "This concept is not in the graph (the documents or the chunk settings changed).")
            sp.set(links=len(detail["links"]), chunks=len(detail["chunks"]), documents=len(detail["documents"]))
        t.set(outcome="shown")
    return detail


# ----------------------------------------------------------------------------- words tab (V2)

def _chunks(corpus, chunk_size, overlap_percent):
    return rag.build_chunks(corpus, chunk_size, overlap_percent)


@app.get("/api/words/vocabulary")
def vocabulary(corpus: str, chunk_size: int = 800, overlap_percent: int = 20, contains: str = "", limit: int = 200):
    return words.vocabulary(_chunks(corpus, chunk_size, overlap_percent), contains, limit)


@app.get("/api/words/word")
def word(corpus: str, word: str, chunk_size: int = 800, overlap_percent: int = 20):
    return words.word_chunks(_chunks(corpus, chunk_size, overlap_percent), word)


@app.post("/api/words/question")
def question_words(body: QuestionWordsIn):
    return {"words": words.question_words(_chunks(body.corpus, body.chunk_size, body.overlap_percent), body.question)}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
