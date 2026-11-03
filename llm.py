"""Gemini through the OpenAI SDK. The answer is validated with Pydantic."""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
NOT_IN_DOCS = "Not in the documents."


class LLMError(Exception):
    """The model call failed or its answer was not valid. The message is safe to show the user."""


class AnswerOut(BaseModel):
    answer: str
    in_documents: bool  # False = the model says the excerpts do not contain the answer


def _client():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set it.")
    return OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def _complete(prompt):
    try:
        r = _client().chat.completions.create(model=MODEL, temperature=0,
                                              messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    except LLMError:
        raise
    except Exception as e:  # network, quota (429), retired model...
        raise LLMError(f"The language model call failed: {e}")


def answer_with_context(question, chunks):
    excerpts = "\n---\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    prompt = f"""You answer questions using ONLY the excerpts below.
If the excerpts do not contain the answer, set "in_documents" to false and answer "{NOT_IN_DOCS}".
Never use outside knowledge. Reply with JSON only, no markdown fence:
{{"answer": "<your answer>", "in_documents": true or false}}

EXCERPTS:
{excerpts}

QUESTION:
{question}"""
    raw = _complete(prompt).strip()
    if raw.startswith("```"):  # models sometimes add a fence anyway
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        out = AnswerOut.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        raise LLMError("The model did not return a valid answer (invalid JSON). Try again.")
    if not out.in_documents:
        out.answer = NOT_IN_DOCS
    return out


def answer_without_context(question):
    """The no-context test: the same question, no retrieval at all."""
    return _complete(f"Answer this question briefly. If you do not know, say so.\n\n{question}").strip()
