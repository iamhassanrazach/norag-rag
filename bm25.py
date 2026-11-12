"""BM25: keyword search that scores rare words higher (a smart Ctrl+F).

For each chunk and each word of the question, the score grows with:
  - how often the word appears in the chunk (tf), with diminishing returns (k1);
  - how RARE the word is in the whole corpus (idf): "KST-2044" is worth much more than "error";
  - and shrinks if the chunk is much longer than average (b).

Where the index lives: a small JSON file in bm25_index/, named like the Chroma collection
(corpus + chunk size + overlap + hash of the documents' content). Change a document and the name
changes, so a fresh index is built; no stale index. It is also cached in memory.
Words are cut by words.tokenize (the same function as the Words tab): lowercase, hyphenated
identifiers kept whole, no stemming; a tiny stop list is removed from the QUESTION only. So what the Words tab shows is exactly what BM25 sees.
"""
import json
import math
import os

import words

BM25_DIR = os.getenv("BM25_DIR", "./bm25_index")
K1, B = 1.5, 0.75  # the usual defaults
_cache = {}


class BM25Index:
    def __init__(self, ids, lengths, postings):
        self.ids = ids                # chunk ids, in order
        self.lengths = lengths        # number of words of each chunk
        self.postings = postings      # word -> [[chunk position, times in that chunk], ...]
        self.avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    @classmethod
    def build(cls, chunks):
        ids, lengths, postings = [], [], {}
        for pos, c in enumerate(chunks):
            tokens = words.tokenize(c["text"])
            ids.append(c["id"])
            lengths.append(len(tokens))
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, n in counts.items():
                postings.setdefault(t, []).append([pos, n])
        return cls(ids, lengths, postings)

    def search(self, question, n=20):
        """Returns [(chunk_id, score), ...], best first. Chunks with no word in common are not returned."""
        scores = {}
        total = len(self.ids)
        # Query words only: drop a tiny generic stop list (what, does, how...). Without it, those words
        # add up and a chunk full of "error" beats the chunk holding the exact code. The index keeps all words.
        for term in {t for t in words.tokenize(question) if t not in words.STOPWORDS}:
            plist = self.postings.get(term)
            if not plist:
                continue  # the word does not exist in the corpus: it contributes nothing
            df = len(plist)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for pos, tf in plist:
                norm = tf + K1 * (1 - B + B * self.lengths[pos] / self.avg_len)
                scores[pos] = scores.get(pos, 0.0) + idf * tf * (K1 + 1) / norm
        ranked = sorted(scores, key=lambda p: (-scores[p], p))[:n]
        return [(self.ids[p], round(scores[p], 4)) for p in ranked]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "lengths": self.lengths, "postings": self.postings}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["ids"], d["lengths"], d["postings"])


def get_index(name, chunks):
    """Memory cache, then file, then build from the chunks (and save)."""
    if name in _cache:
        return _cache[name]
    path = os.path.join(BM25_DIR, f"{name}.json")
    index = None
    if os.path.exists(path):
        try:
            index = BM25Index.load(path)
            if index.ids != [c["id"] for c in chunks]:  # file does not match the chunks: rebuild
                index = None
        except (OSError, ValueError, KeyError):
            index = None
    if index is None:
        index = BM25Index.build(chunks)
        index.save(path)
    _cache[name] = index
    return index
