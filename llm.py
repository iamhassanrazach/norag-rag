"""Gemini through the OpenAI SDK. The answer is validated with Pydantic."""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
NOT_IN_DOCS = "Not in the documents."


class LLMError(Exception):
    """The model call failed or its answer was not valid. The message is safe to show the user."""


class AnswerOut(BaseModel):
    answer: str
    in_documents: bool  # False = the model says the excerpts do not contain the answer


STATS = {"calls": 0, "tokens": 0}  # counted for the cost columns of the evaluation and of the graph build
_key_i = 0


def _keys():
    """GOOGLE_API_KEYS="k1,k2,k3" (rotation, helps with the free-tier rate limit) or the single GOOGLE_API_KEY."""
    keys = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
    if not keys and os.getenv("GOOGLE_API_KEY"):
        keys = [os.getenv("GOOGLE_API_KEY")]
    if not keys:
        raise LLMError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set it.")
    return keys


def _complete(prompt, model=None, retries=4):
    """One model call. On a 429 (rate limit) switch to the next key and wait a little longer each time."""
    global _key_i
    keys = _keys()
    last = None
    for attempt in range(retries + 1):
        key = keys[_key_i % len(keys)]
        try:
            client = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            r = client.chat.completions.create(model=model or MODEL, temperature=0,
                                               messages=[{"role": "user", "content": prompt}])
            STATS["calls"] += 1
            STATS["tokens"] += getattr(r.usage, "total_tokens", 0) or 0
            return r.choices[0].message.content or ""
        except RateLimitError as e:
            last = e
            _key_i += 1
            time.sleep(min(2 ** attempt * 2, 30))
        except Exception as e:  # network, retired model...
            raise LLMError(f"The language model call failed: {e}")
    raise LLMError(f"The language model call failed (rate limit, {retries + 1} tries): {last}")


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
