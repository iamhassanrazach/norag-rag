"""Repeatable evaluation of the retrieval (and, on demand, of the answers).

    python evaluate.py kestrel                                  # retrieval only: fast and free
    python evaluate.py kestrel --chunk-size 400 --overlap 20    # another configuration
    python evaluate.py kestrel --retrievers dense,bm25          # V3: which retrievers are switched on
    python evaluate.py kestrel --with-answers                   # also generate answers (LLM calls)
    python evaluate.py --table results/a.json results/b.json    # compare runs (markdown table for the report)
    python evaluate.py --grades results/<file>.json             # count your hand grades

A retrieved chunk is RELEVANT to an evidence item if it comes from the evidence's source file
AND contains its quote (whitespace and case ignored).
  HitRate@k = share of questions with at least one relevant chunk in the top k.
  Recall@k  = average share of a question's evidence items found in the top k.
Unanswerable questions have no evidence: they are left out of both measures.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import rag

KS = [1, 3, 5, 10]
RESULTS_DIR = "results"


def norm(s):
    return " ".join(s.split()).lower()


def matches(chunk, evidence):
    """Indexes of the evidence items this chunk supports."""
    return [i for i, e in enumerate(evidence)
            if chunk["source"] == e["source"] and norm(e["quote"]) in norm(chunk["text"])]


def average(values):
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows):
    """rows: per-question dicts with type and metrics {k: {hit, recall}} (answerable questions only)."""
    def block(rs):
        return {"n": len(rs), **{str(k): {"hit_rate": average([r["metrics"][str(k)]["hit"] for r in rs]),
                                           "recall": average([r["metrics"][str(k)]["recall"] for r in rs])}
                                 for k in KS}}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"overall": block(rows), "by_type": {t: block(rs) for t, rs in sorted(by_type.items())}}


def cost_summary(rows):
    n = len(rows) or 1
    return {"retrieval_ms": round(sum(r["cost"]["retrieval_ms"] for r in rows) / n, 1),
            "llm_calls": round(sum(r["cost"]["llm_calls"] for r in rows) / n, 2),
            "tokens": round(sum(r["cost"]["tokens"] for r in rows) / n, 1),
            "answer_ms": round(sum(r["cost"]["answer_ms"] for r in rows) / n, 1)}


def run_eval(corpus, chunk_size=800, overlap_percent=10, with_answers=False, answer_k=5, questions=None,
             retrievers=("dense",), candidates=20, rrf_k=60):
    retrievers = tuple(retrievers)
    questions = questions if questions is not None else rag.load_questions(corpus)
    if not questions:
        raise rag.CorpusError(f"Corpus '{corpus}' has no questions (corpora/{corpus}/questions.jsonl).")
    all_chunks = rag.build_chunks(corpus, chunk_size, overlap_percent)
    rag.prepare(corpus, chunk_size, overlap_percent, retrievers)  # one-off index builds are not timed

    results, scored, unreachable = [], [], []
    for q in questions:
        evidence = q.get("evidence", [])
        # Sanity check: is each quote complete inside at least one chunk of this configuration?
        # If not (cut by a chunk boundary), no retrieval can ever count it.
        for i, e in enumerate(evidence):
            if not any(i in matches(c, evidence) for c in all_chunks):
                unreachable.append({"id": q["id"], "source": e["source"], "quote": e["quote"]})

        t0 = time.perf_counter()
        retrieved = rag.search(corpus, q["question"], k=max(KS), chunk_size=chunk_size,
                               overlap_percent=overlap_percent, retrievers=retrievers,
                               candidates=candidates, rrf_k=rrf_k)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        for c in retrieved:
            c["relevant"] = bool(matches(c, evidence))
        row = {"id": q["id"], "question": q["question"], "type": q["type"], "answerable": q["answerable"],
               "retrieved": retrieved, "metrics": None, "grade": None, "failure_kind": None,
               "cost": {"retrieval_ms": retrieval_ms, "llm_calls": 0, "tokens": 0, "answer_ms": 0}}

        if q["answerable"] and evidence:
            row["metrics"] = {}
            for k in KS:
                found = set()
                for c in retrieved[:k]:
                    found.update(matches(c, evidence))
                row["metrics"][str(k)] = {"hit": int(bool(found)), "recall": round(len(found) / len(evidence), 3)}
            scored.append(row)

        if with_answers:
            import llm
            calls0, tokens0, t1 = llm.STATS["calls"], llm.STATS["tokens"], time.perf_counter()
            try:
                a = llm.answer_with_context(q["question"], retrieved[:answer_k])
                row["answer"] = {"text": a.answer, "in_documents": a.in_documents, "used_top_k": answer_k}
            except llm.LLMError as e:
                row["answer"] = {"error": str(e)}
            row["cost"].update(llm_calls=llm.STATS["calls"] - calls0, tokens=llm.STATS["tokens"] - tokens0,
                               answer_ms=round((time.perf_counter() - t1) * 1000, 1))
        results.append(row)

    return {
        "meta": {"corpus": corpus, "chunk_size": chunk_size, "overlap_percent": overlap_percent,
                 "retrievers": list(retrievers), "candidates": candidates, "rrf_k": rrf_k, "ks": KS, "with_answers": with_answers, "answer_k": answer_k if with_answers else None,
                 "n_chunks": len(all_chunks), "n_questions": len(questions), "n_scored": len(scored),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "summary": {**summarize(scored), "cost_per_question": cost_summary(results)},
        "unreachable_evidence": unreachable,
        "questions": results,
    }


def save(result):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    m = result["meta"]
    path = os.path.join(RESULTS_DIR, f"{m['corpus']}__{'+'.join(m['retrievers'])}__cs{m['chunk_size']}_ov{m['overlap_percent']}"
                                     f"{'_answers' if m['with_answers'] else ''}__{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    return path


def print_summary(result):
    m, s = result["meta"], result["summary"]
    print(f"\ncorpus={m['corpus']} retrievers={'+'.join(m['retrievers'])} chunk_size={m['chunk_size']} "
          f"overlap={m['overlap_percent']}% ({m['n_chunks']} chunks, {m['n_scored']} scored questions)")
    print(f"{'':20}{'n':>3}  " + "  ".join(f"hit@{k:<2} rec@{k:<2}" for k in KS))
    for name, b in [("OVERALL", s["overall"])] + list(s["by_type"].items()):
        print(f"{name:20}{b['n']:>3}  " + "  ".join(f"{b[str(k)]['hit_rate']:<6} {b[str(k)]['recall']:<6}" for k in KS))
    c = s["cost_per_question"]
    print(f"\ncost per question: retrieval {c['retrieval_ms']} ms, LLM calls {c['llm_calls']}, "
          f"tokens {c['tokens']}, answer time {c['answer_ms']} ms")
    if result["unreachable_evidence"]:
        print(f"\nWARNING: {len(result['unreachable_evidence'])} evidence quote(s) are cut by a chunk boundary "
              f"in this configuration (no retrieval can match them): see 'unreachable_evidence' in the file.")


def print_table(paths):
    """Markdown table (one row per run and question type) to paste into REPORT.md."""
    head = ["corpus", "retrievers", "chunk/overlap", "type", "n"] + [f"{m}@{k}" for k in KS for m in ("hit", "rec")]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        m, s = d["meta"], d["summary"]
        for name, b in [("OVERALL", s["overall"])] + list(s["by_type"].items()):
            cells = [m["corpus"], "+".join(m.get("retrievers", ["dense"])),
                     f"{m['chunk_size']}/{m['overlap_percent']}", name, str(b["n"])]
            cells += [str(b[str(k)][key]) for k in KS for key in ("hit_rate", "recall")]
            print("| " + " | ".join(cells) + " |")


def print_grades(path):
    data = json.load(open(path, encoding="utf-8"))
    by_type = defaultdict(Counter)
    for q in data["questions"]:
        by_type[q["type"]][q["grade"] or "ungraded"] += 1
    for t, c in sorted(by_type.items()):
        print(f"{t:18}", dict(c))
    print("TOTAL", dict(sum(by_type.values(), Counter())))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("corpus", nargs="?", help="folder name under corpora/ (kestrel, own...)")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=10, help="overlap in percent")
    p.add_argument("--with-answers", action="store_true", help="also generate answers (LLM calls)")
    p.add_argument("--answer-k", type=int, default=5, help="how many chunks the answer is generated from")
    p.add_argument("--retrievers", default="dense", help="comma list among: dense,bm25 (default dense = V2)")
    p.add_argument("--candidates", type=int, default=20, help="how many chunks each retriever hands to the fusion")
    p.add_argument("--rrf-k", type=int, default=60, help="the k of RRF: score = 1 / (k + rank)")
    p.add_argument("--grades", metavar="FILE", help="print the hand grades of a results file")
    p.add_argument("--table", nargs="+", metavar="FILE", help="markdown comparison table of results files")
    a = p.parse_args()
    if a.table:
        print_table(a.table)
    elif a.grades:
        print_grades(a.grades)
    elif a.corpus:
        res = run_eval(a.corpus, a.chunk_size, a.overlap, a.with_answers, a.answer_k,
                       retrievers=a.retrievers.split(","), candidates=a.candidates, rrf_k=a.rrf_k)
        print_summary(res)
        print("\nSaved:", save(res))
    else:
        p.print_help()
