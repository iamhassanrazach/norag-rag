"""Web API + front-end.  Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import evaluate
import llm
import rag
import words

app = FastAPI(title="RAG V2")


class Config(BaseModel):
    corpus: str
    chunk_size: int = Field(800, ge=100, le=4000)
    overlap_percent: int = Field(10, ge=0, le=50)


class SearchIn(Config):
    question: str
    k: int = Field(5, ge=1, le=10)


class EvalIn(Config):
    with_answers: bool = False
    answer_k: int = Field(5, ge=1, le=10)


class QuestionWordsIn(Config):
    question: str


@app.exception_handler(rag.CorpusError)
async def corpus_error(_, e):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(e)})


@app.exception_handler(llm.LLMError)
async def llm_error(_, e):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=502, content={"detail": str(e)})


@app.get("/api/corpora")
def corpora():
    out = []
    for c in rag.list_corpora():
        try:
            n_docs = len(rag.read_documents(c))
        except rag.CorpusError:
            n_docs = 0
        out.append({"name": c, "documents": n_docs, "questions": len(rag.load_questions(c))})
    return out


@app.get("/api/questions")
def questions(corpus: str):
    return rag.load_questions(corpus)


@app.post("/api/search")
def search(body: SearchIn):
    return {"chunks": rag.search(body.corpus, body.question, body.k, body.chunk_size, body.overlap_percent)}


@app.post("/api/ask")
def ask(body: SearchIn):
    chunks = rag.search(body.corpus, body.question, body.k, body.chunk_size, body.overlap_percent)
    a = llm.answer_with_context(body.question, chunks)
    return {"answer": a.answer, "in_documents": a.in_documents, "chunks": chunks}


@app.post("/api/eval/run")
def run_eval(body: EvalIn):
    result = evaluate.run_eval(body.corpus, body.chunk_size, body.overlap_percent, body.with_answers, body.answer_k)
    result["saved_to"] = evaluate.save(result)
    return result


def _chunks(corpus, chunk_size, overlap_percent):
    return rag.build_chunks(corpus, chunk_size, overlap_percent)


@app.get("/api/words/vocabulary")
def vocabulary(corpus: str, chunk_size: int = 800, overlap_percent: int = 10, contains: str = "", limit: int = 200):
    return words.vocabulary(_chunks(corpus, chunk_size, overlap_percent), contains, limit)


@app.get("/api/words/word")
def word(corpus: str, word: str, chunk_size: int = 800, overlap_percent: int = 10):
    return words.word_chunks(_chunks(corpus, chunk_size, overlap_percent), word)


@app.post("/api/words/question")
def question_words(body: QuestionWordsIn):
    return {"words": words.question_words(_chunks(body.corpus, body.chunk_size, body.overlap_percent), body.question)}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
