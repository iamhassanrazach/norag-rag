"""The no-context test: ask every question to the LLM WITHOUT retrieval.

    python no_context_test.py kestrel
    python no_context_test.py own

Saves results/no_context_<corpus>.json. Then read each model answer next to expected_answer
and set "model_correct" to true or false BY HAND. Questions the model gets right without
the documents do not test your RAG: replace them, or keep them and say so in REPORT.md.
"""
import json
import os
import sys

import llm
import rag

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    corpus = sys.argv[1]
    rows = []
    for q in rag.load_questions(corpus):
        try:
            answer = llm.answer_without_context(q["question"])
        except llm.LLMError as e:
            answer = f"ERROR: {e}"
        print(f"\n[{q['id']}] {q['question']}\n  model : {answer[:300]}\n  expect: {q.get('expected_answer')}")
        rows.append({"id": q["id"], "question": q["question"], "type": q["type"],
                     "expected_answer": q.get("expected_answer"), "model_answer": answer, "model_correct": None})
    os.makedirs("results", exist_ok=True)
    path = f"results/no_context_{corpus}.json"
    json.dump({"model": llm.MODEL, "questions": rows}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nSaved:", path, '-> now set "model_correct" by hand.')
