# RAG project - V2

Web app to ask questions about a document corpus (answers with references to the chunks used),
plus a repeatable evaluation and a Words tab. The original course README is in `README_BASE.md`.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env     # then put your GOOGLE_API_KEY in .env (remove the word "export" if you like)
```

## Run the app and reach the Words tab

```powershell
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. Three tabs: **Ask**, **Evaluation**, **Words**. The corpus, chunk size,
overlap and top-k are chosen in the bar at the top. The first search on a new configuration builds its
index (a few seconds for Kestrel).

## Corpora and question sets

```
corpora/kestrel/docs/ + questions.jsonl   provided Kestrel KB + its 10 starter questions
corpora/own/docs/     + questions.jsonl   OUR corpus and OUR questions (same JSONL format)
```

Question sets stay fixed while comparing configurations: add questions, never edit old ones silently.

## Run the evaluation

From the terminal (one command; results are saved in `results/`):

```powershell
python evaluate.py kestrel                              # retrieval only, fast and free
python evaluate.py kestrel --chunk-size 400 --overlap 20
python evaluate.py own
python evaluate.py kestrel --with-answers --answer-k 5  # also generates answers (LLM calls)
python evaluate.py --grades results\<file>.json         # count the hand grades
```

From the front-end: **Evaluation** tab, optional checkbox "also generate answers", button "Run evaluation".
Both do the same thing (`evaluate.run_eval`).

Measures: **HitRate@k** and **Recall@k** for k = 1, 3, 5, 10, overall and per question type. A chunk is relevant
if it comes from the evidence's source file and contains its quote. Unanswerable questions are left out.

## No-context test

```powershell
python no_context_test.py kestrel
python no_context_test.py own
```

Saves `results/no_context_<corpus>.json`; set `model_correct` by hand for each question.

## Hand grades

Open an `..._answers_...json` results file and fill `grade` (`correct`, `partial`, `wrong`, `unsupported`) and, for
wrong answers, `failure_kind` (`retrieval` or `generation`). Grading rules are written in REPORT.md before grading.

## Tests

```powershell
python -m pytest -q tests     # uses a fake embedding, no model download, no API key
```
