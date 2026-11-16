# RAG project - V4 (final)

Ask questions about your own documents. Every answer shows the passages it used, and every question and every
indexing run leaves a **trace** you can open from the answer. The original course README is in `README_BASE.md`.

## Install and run (Windows PowerShell, 5 minutes)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then open .env and fill the three LLM_ lines (see below)
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. The first question downloads a small embedding model (a minute, once).

## Choose the AI model (`.env`)

```
LLM_URL=...        # address of any OpenAI-compatible server. Empty = Google Gemini (default)
LLM_API_KEY=...    # your key (several keys separated by commas are rotated when one hits the rate limit)
LLM_MODEL=...      # model name, e.g. gemini-2.5-flash-lite or a local model
```

Nothing else in the code needs to change to use another provider. A local server (Ollama, LM Studio...) needs
no key. The old names `GOOGLE_API_KEY(S)` and `GEMINI_MODEL` still work. The cost shown in traces is an estimate
from `LLM_PRICE_IN` / `LLM_PRICE_OUT` (USD per million tokens; check them for your model).

## What you can do in the app

| Tab | What it is for |
|---|---|
| **Ask** | Ask a question. The answer lists its source passages and which search found each. **Details** opens the trace. |
| **My documents** | Create a collection, drop `.md` / `.txt` files in it (max 2 MB each, 100 files), press **Prepare**. Tick "build the concept map" to also extract the concept graph (uses the LLM; only new passages cost calls). The provided collections (`kestrel`, `own`) are read-only. |
| **Concept map** | Explore the concept graph of a collection: click a concept to see its links, the quote each link comes from, and the passages. "Ask a question about it" jumps to Ask. Building the map is the only step that uses the LLM, opening it costs nothing. |
| **History** | Counts, time per step (median and 95th percentile), tokens and cost per day, and the list of recent requests. |
| Evaluation, Words | As in V2/V3 (see below). |

Tip: to try the graph on a small collection first (saves quota), upload 2-3 short files and build it there.

Two graph schemas exist: the support schema (Service, Team, ErrorCode...) is used for Kestrel; every other
collection uses a general one (Concept, Person, Organization, Place, Product, Event).

## Traces (observability)

Each question and each indexing run writes one trace: a tree of steps (spans) with duration, status and details
(retrieval per retriever, fusion, prompt build, each LLM call with model, tokens, finish reason and cost, validation).
The final outcome is one of: answered, abstained, invalid_output, error (questions); indexed, partial, error (indexing).

- Stored in `traces/YYYY-MM-DD.jsonl`, one file per day, **on this computer only**, never sent anywhere.
- Kept 30 days (`TRACE_RETENTION_DAYS`); older files are deleted at start-up.
- API keys are removed before writing; personal data is hidden (next section); only passage ids are stored, not passage text.
- Same data via the API: `GET /api/traces`, `/api/traces/{id}`, `/api/metrics?days=7`.

## Privacy: personal data hidden before it is sent

On by default (`PII_REDACTION=0` turns it off). Before the question and the passages go to the AI provider, emails, phone
numbers, card numbers (Luhn check), IBANs (checksum) and names after a title ("Mr Smith") become placeholders such as
`[EMAIL_1]`. The provider and the traces only see placeholders; the answer shown to you has the real values put back.
Names you always want hidden can be listed in `PII_NAMES="Jane Doe,John Roe"`.
**Not covered:** bare names, postal addresses, dates of birth, IP addresses, national ID numbers. Rules, not magic.

Prompt injection: passages are given to the model as data (inside `<excerpts>` tags) with an instruction not to follow
orders found in them (prompt version `answer-v4`).

## Tests

```powershell
python -m pytest -q       # about 50 tests, fake LLM server and fake embeddings: no key, no download, no quota
python tests\fake_llm.py  # optional: a fake AI server on port 8765 to try the app offline (LLM_URL=http://127.0.0.1:8765)
```

---

## V2 and V3 (kept)

### Corpora and question sets

```
corpora/kestrel/docs/ + questions.jsonl   provided Kestrel KB + its 10 starter questions
corpora/own/docs/     + questions.jsonl   OUR corpus and OUR questions
user_corpora/<name>/                      collections created in the app (not in Git)
```

### Run the evaluation

```powershell
python evaluate.py kestrel                              # retrieval only, fast and free
python evaluate.py kestrel --chunk-size 400 --overlap 20
python evaluate.py own
python evaluate.py kestrel --with-answers --answer-k 5  # also generates answers (LLM calls)
python evaluate.py --grades results\<file>.json         # count the hand grades
```

Measures: **HitRate@k** and **Recall@k** for k = 1, 3, 5, 10. A chunk is relevant if it comes from the evidence's source
file and contains its quote. Unanswerable questions are left out. Hand grades: open an `..._answers_...json` file and fill
`grade` (`correct`, `partial`, `wrong`, `unsupported`) and, for wrong answers, `failure_kind` (`retrieval` or `generation`).

### No-context test

```powershell
python no_context_test.py kestrel
python no_context_test.py own
```

Saves `results/no_context_<corpus>.json`; set `model_correct` by hand for each question.

### Hand grades

Open an `..._answers_...json` results file and fill `grade` (`correct`, `partial`, `wrong`, `unsupported`) and, for
wrong answers, `failure_kind` (`retrieval` or `generation`). Grading rules are written in REPORT.md before grading.

### Tests

```powershell
python -m pytest -q tests     # uses a fake embedding, no model download, no API key
```

---

## V3: retrievers, fusion, evaluation

Three ways to find chunks, each switched on or off, then merged by RRF:

| Retriever | Idea | Status |
|---|---|---|
| `dense` | embeddings (meaning), as in V2 | done |
| `bm25` | keywords scored by rarity (own implementation, `bm25.py`) | done |
| `graph` | concept graph built by the LLM (Kestrel), walked without any LLM call | done (`graph.py`) |

**Switch them in the UI:** the *Search* menu on the Ask tab (meaning / meaning + exact words / + concept map); the Evaluation tab has its own menu.
`Candidates` is how many chunks each retriever hands to the fusion, `RRF k` is the constant of
score = 1 / (k + rank). In the Ask tab each chunk shows which retriever found it, at which rank.

**Switch them in the terminal:**

```powershell
python evaluate.py kestrel --retrievers dense                  # = V2
python evaluate.py kestrel --retrievers bm25
python evaluate.py kestrel --retrievers dense,bm25             # hybrid (RRF)
python evaluate.py own     --retrievers dense,bm25
python evaluate.py kestrel --retrievers dense,bm25 --candidates 30 --rrf-k 60
python evaluate.py --table results\a.json results\b.json      # markdown table to paste in REPORT.md
```

Results files are named `<corpus>__<retrievers>__cs<size>_ov<overlap>__<time>.json` and also contain, per question,
`found_by` (retriever -> rank) for every retrieved chunk, and the cost (retrieval ms, LLM calls, tokens, answer ms).

BM25 index: JSON file in `bm25_index/` (ignored by Git), rebuilt automatically when a document, the chunk size or the
overlap changes. See the top of `bm25.py` for the tokenization.

### Concept graph (`graph.py`)

The LLM reads **every chunk once** and writes (entity, relation, entity) triples. Nothing is asked of the LLM at query time.

```powershell
python graph.py build kestrel --chunk-size 800 --overlap 20     # ~47 Gemini calls; cached in graph_index\, safe to stop and restart
python graph.py stats kestrel --chunk-size 800 --overlap 20     # nodes, edges, build cost (calls, tokens, seconds)
python graph.py ask   kestrel "Which team owns the service that raises KST-2044?" --chunk-size 800 --overlap 20
python graph.py sample kestrel --chunk-size 800 --overlap 20    # writes results\graph_triples_kestrel.json (10 random triples)
python graph.py grades results\graph_triples_kestrel.json       # counts your correct / wrong / unsupported
python evaluate.py kestrel --retrievers dense,bm25,graph --chunk-size 800 --overlap 20
python evaluate.py kestrel --retrievers dense,bm25,graph --weights graph=2 --candidates 30   # give the graph more say
```

Rate limit (429): put several keys in `.env` as `GOOGLE_API_KEYS="key1,key2"` (the code rotates and waits), or use `--pause 4`.
The graph is **chunk-size specific**: it is built for one chunk size/overlap and the cache name contains a hash of the documents.

**Schema (fixed).** Entities: Service, Team, ErrorCode, Plan, Feature, Policy, Channel.
Relations: OWNED_BY, ESCALATED_TO, REPLACES, DEPENDS_ON, RAISED_BY, AVAILABLE_ON, GOVERNED_BY.
A triple outside the schema is dropped (counted in `dropped_triples`).

**Entity resolution (simple).** Same type + same normalised name (lower case, without "the", trailing "service" / "team" / "plan"),
or one name listed as an alias by the LLM. Known limits: two different things with the same name merge; one thing with two
names that the LLM did not list as aliases stays split. Look at `python graph.py stats` and the sample to find real examples.

**Retrieval (no LLM).** Nodes whose name appears in the question (word-boundary match) are the *seeds*. The seed's own chunks
score 1; walking 1 or 2 hops in the graph, each edge adds 1/(distance+1) to the chunks that state it. The ranked chunks go into RRF
like any other retriever. In the Ask tab each graph chunk shows its **path** (the edges that led to it).

**Cost.** One build (calls / tokens / seconds, in the cache file and in `graph.py stats`) and then 0 LLM calls per question.
Re-running `build` after the cache exists costs nothing; changing `PROMPT_VERSION` in `graph.py` forces a new extraction.
