"""Words analytics: which words does the corpus contain, and in which chunks?

Deliberately simple: lowercase, split on letters/digits, keep hyphenated
identifiers together (KST-2044 stays ONE word). No stemming, no synonyms:
a word "exists" only if it is written exactly like that in a chunk.
"""
import re
from collections import Counter

WORD_RE = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)

# Tiny stop list, only used to grey out words in question mode.
STOPWORDS = set("""a an the of to in on at for and or but is are was were be been do does did can could
should would what which who how when where why i my me we our you your it its this that these those
with without from by as if then so there no not any anyone""".split())


def tokenize(text):
    return WORD_RE.findall(text.lower())


def count_per_chunk(chunks):
    """chunks: list of {id, source, chunk_index, text}. Returns {chunk_id: Counter}."""
    return {c["id"]: Counter(tokenize(c["text"])) for c in chunks}


def vocabulary(chunks, contains="", limit=200):
    per_chunk = count_per_chunk(chunks)
    total, n_chunks = Counter(), Counter()
    for counter in per_chunk.values():
        total.update(counter)
        n_chunks.update(counter.keys())  # +1 per chunk containing the word
    rows = [{"word": w, "count": total[w], "chunks": n_chunks[w]}
            for w in total if contains.lower() in w]
    rows.sort(key=lambda r: (-r["count"], r["word"]))
    return {"distinct_words": len(total), "total_words": sum(total.values()),
            "shown": rows[:limit]}


def word_chunks(chunks, word):
    word = word.lower().strip()
    per_chunk = count_per_chunk(chunks)
    out = []
    for c in chunks:
        n = per_chunk[c["id"]][word]
        if n:
            out.append({"id": c["id"], "source": c["source"], "chunk_index": c["chunk_index"],
                        "count": n, "text": c["text"]})
    out.sort(key=lambda r: -r["count"])
    return {"word": word, "total": sum(r["count"] for r in out), "chunks": out}


def question_words(chunks, question):
    seen, rows = set(), []
    for w in tokenize(question):
        if w in seen:
            continue
        seen.add(w)
        found = word_chunks(chunks, w)
        rows.append({"word": w, "stopword": w in STOPWORDS, "exists": found["total"] > 0,
                     "total": found["total"], "n_chunks": len(found["chunks"]),
                     "chunks": [{"id": c["id"], "source": c["source"], "count": c["count"]}
                                for c in found["chunks"]]})
    return rows
