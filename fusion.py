"""Reciprocal Rank Fusion (RRF): merge ranked lists from several retrievers.

Each chunk gets, from every list it appears in, a score of  weight / (k + rank).
The scores are added and the chunks re-sorted. Only RANKS count (not distances or BM25 scores,
which cannot be compared). k = 60 is the usual value: the bigger k, the flatter the curve.
"""


def rrf(ranked_lists, k=60, weights=None, top=10):
    """ranked_lists: {retriever name: [chunk dict with 'id', best first]}. Returns the fused chunks,
    each with 'rrf_score' and 'found_by' = {retriever name: its rank for that chunk}."""
    weights = weights or {}
    order = {name: i for i, name in enumerate(ranked_lists)}
    scores, found, merged = {}, {}, {}
    for name, chunks in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, c in enumerate(chunks, start=1):
            cid = c["id"]
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            found.setdefault(cid, {})[name] = rank
            m = merged.setdefault(cid, dict(c))
            for key, value in c.items():          # keep distance (dense) AND bm25_score (bm25)
                m.setdefault(key, value)

    def sort_key(cid):  # equal scores: best single rank first, then retriever order, then id
        f = found[cid]
        return (-scores[cid], min(f.values()), min(order[n] for n in f), cid)

    out = []
    for rank, cid in enumerate(sorted(scores, key=sort_key)[:top], start=1):
        c = merged[cid]
        c.update(rank=rank, rrf_score=round(scores[cid], 5), found_by=found[cid])
        out.append(c)
    return out
