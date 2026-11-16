"""Concept graph: the LLM reads every chunk once and writes down (entity, relation, entity) triples.

    python graph.py build kestrel            # one Gemini call per chunk (cached on disk, so a re-run is free)
    python graph.py stats kestrel            # nodes, edges, build cost
    python graph.py sample kestrel           # 10 random triples to grade by hand (correct / wrong / unsupported)
    python graph.py grades results/graph_triples_kestrel.json
    python graph.py debug kestrel "service-catalog.md::2"   # show the raw model answer for one chunk
    python graph.py ask kestrel "Who owns the service that raises KST-2044?"   # see seeds, edges followed, chunks

How it works (the four design decisions):
  1. FIXED SCHEMA: 7 entity types and 7 relation types (below). A triple outside the schema is dropped, not repaired.
  2. SIMPLE ENTITY RESOLUTION: two mentions are the same node if they have the same type and the same normalised
     name (lower case, no "the", no trailing "service"/"team"), or if one lists the other as an alias.
     Wrong merges and missed merges are possible: that is a documented limit, not hidden.
  3. NO LLM AT QUERY TIME: the question is matched to node names by plain string matching (word boundaries),
     then the graph is walked 1-2 hops. The cost of the graph is paid once, at build time.
  4. PROVENANCE: every node and every edge remembers the chunk ids (and the quote) it came from.
     The retriever returns CHUNKS, ranked, so it can be fused with dense and BM25 by RRF.
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict, deque

from pydantic import BaseModel, ValidationError

import llm
import pii
import tracing

GRAPH_DIR = os.getenv("GRAPH_DIR", "graph_index")
RESULTS_DIR = "results"

ENTITY_TYPES = ("Service", "Team", "ErrorCode", "Plan", "Feature", "Policy", "Channel")
RELATIONS = {
    "OWNED_BY":     "Service -> Team (the team that owns / is on call for the service)",
    "ESCALATED_TO": "ErrorCode, Service or Feature -> Team or Channel (where to escalate)",
    "REPLACES":     "X -> Y of the same type (X replaced Y: old service, superseded policy)",
    "DEPENDS_ON":   "Service or Feature -> Service (needs it, runs through it, is sent by it)",
    "RAISED_BY":    "ErrorCode -> Service (the service that produces the error)",
    "AVAILABLE_ON": "Feature -> Plan (the plan includes the feature)",
    "GOVERNED_BY":  "Plan, Feature or ErrorCode -> Policy (the policy that applies)",
}
# allowed (subject types, object types) per relation: a triple with the wrong direction is dropped, not repaired
REL_TYPES = {
    "OWNED_BY":     ({"Service"}, {"Team"}),
    "ESCALATED_TO": ({"ErrorCode", "Service", "Feature"}, {"Team", "Channel"}),
    "REPLACES":     (set(ENTITY_TYPES), set(ENTITY_TYPES)),   # same type on both sides (checked below)
    "DEPENDS_ON":   ({"Service", "Feature"}, {"Service"}),
    "RAISED_BY":    ({"ErrorCode"}, {"Service"}),
    "AVAILABLE_ON": ({"Feature"}, {"Plan"}),
    "GOVERNED_BY":  ({"Plan", "Feature", "ErrorCode"}, {"Policy"}),
}
MAX_ASSIGNED_HOPS = 2
PROMPT_VERSION = 3  # change the prompt => change this number => the cache is not reused

# Two schemas ("profiles"). The V3 one is designed for the Kestrel support knowledge base and stays untouched (our V3
# measurements depend on it). Any OTHER corpus (documents uploaded from the UI) uses the general one: broader types,
# fewer constraints. Choosing a profile is automatic: profile_for(corpus).
GENERAL_TYPES = ("Concept", "Person", "Organization", "Place", "Product", "Event")
GENERAL_RELATIONS = {
    "PART_OF":    "X -> Y (X is a part, component, version or member of Y)",
    "CREATED_BY": "X -> Person or Organization (made, built, founded, written or decided by)",
    "LOCATED_IN": "X -> Place (where X is, is made, or happened)",
    "REPLACES":   "X -> Y of the same type (X replaced or succeeded Y)",
    "USES":       "X -> Y (X uses, needs, contains or depends on Y)",
    "RELATED_TO": "X -> Y (any other clear relationship that the text states in so many words)",
}
SUPPORT = {"name": "support", "version": PROMPT_VERSION, "types": ENTITY_TYPES, "relations": RELATIONS, "rel_types": REL_TYPES}
GENERAL = {"name": "general", "version": 1, "types": GENERAL_TYPES, "relations": GENERAL_RELATIONS, "rel_types": {}}
SUPPORT_CORPORA = {"kestrel"}


def profile_for(corpus):
    return SUPPORT if corpus in SUPPORT_CORPORA else GENERAL


class Triple(BaseModel):
    subject: str
    subject_type: str
    subject_aliases: list[str] = []
    relation: str
    object: str
    object_type: str
    object_aliases: list[str] = []
    evidence: str  # a short quote from the chunk that supports the triple


class Extraction(BaseModel):
    triples: list[Triple]


def _prompt(chunk, profile=SUPPORT):
    if profile["name"] == "general":
        return _general_prompt(chunk)
    rels = "\n".join(f"  {r}: {d}" for r, d in RELATIONS.items())
    return f"""Extract a knowledge graph from the text below. Use ONLY facts written in the text.

Entity types (use exactly these): {", ".join(ENTITY_TYPES)}
Relations (use exactly these):
{rels}

Rules:
- Be THOROUGH: write one triple for every relationship of these types that the text states, including ones in tables,
  headings and "Owner: X" lines. A heading names the entity that the paragraph below it describes.
- Direction matters: the subject comes first, as in the relation list. "Owner: X" under a service means
  <service> OWNED_BY <X> (subject = the Service, object = the Team), never the other way round.
- Only write a triple if the text states it. Never guess. Return an empty list only if the text really states none.
- Use the name as written in the text (e.g. "KST-2044", "postroom", "Analytics Engine", "Pro plan").
- subject_aliases / object_aliases: other names the SAME text uses for the SAME thing (usually empty).
- evidence: a short quote copied from the text that supports the triple.
Example (made-up text, NOT from this corpus):
  TEXT: "### invoice-api\nCreates invoices. Owner: **Payments**. Errors XYZ-1 come from invoice-api; see the refund policy."
  ANSWER: {{"triples": [
    {{"subject": "invoice-api", "subject_type": "Service", "subject_aliases": [], "relation": "OWNED_BY",
      "object": "Payments", "object_type": "Team", "object_aliases": [], "evidence": "Owner: **Payments**"}},
    {{"subject": "XYZ-1", "subject_type": "ErrorCode", "subject_aliases": [], "relation": "RAISED_BY",
      "object": "invoice-api", "object_type": "Service", "object_aliases": [], "evidence": "Errors XYZ-1 come from invoice-api"}}]}}

Reply with JSON only, no markdown fence:
{{"triples": [{{"subject": "...", "subject_type": "...", "subject_aliases": [], "relation": "...",
  "object": "...", "object_type": "...", "object_aliases": [], "evidence": "..."}}]}}

TEXT (from {chunk['source']}):
{chunk['text']}"""


def _general_prompt(chunk):
    rels = "\n".join(f"  {r}: {d}" for r, d in GENERAL_RELATIONS.items())
    return f"""Extract a knowledge graph from the text below. Use ONLY facts written in the text.

Entity types (use exactly these): {", ".join(GENERAL_TYPES)}
Relations (use exactly these):
{rels}

Rules:
- Write one triple for every relationship of these types that the text states, including ones in tables and headings.
- The subject comes first, as in the relation list. Use the most specific relation; RELATED_TO only when no other fits.
- Only write a triple if the text states it. Never guess. Return an empty list only if the text really states none.
- Use the name as written in the text, short and without articles ("Aventador", "V12 engine", "Sant'Agata Bolognese").
- subject_aliases / object_aliases: other names the SAME text uses for the SAME thing (usually empty).
- evidence: a short quote copied from the text that supports the triple.
Example (made-up text): "The Zeta 3 replaced the Zeta 2 in 2021. Its battery pack is built by Voltix in Lyon."
  ANSWER: {{"triples": [
    {{"subject": "Zeta 3", "subject_type": "Product", "subject_aliases": [], "relation": "REPLACES",
      "object": "Zeta 2", "object_type": "Product", "object_aliases": [], "evidence": "The Zeta 3 replaced the Zeta 2"}},
    {{"subject": "battery pack", "subject_type": "Concept", "subject_aliases": [], "relation": "CREATED_BY",
      "object": "Voltix", "object_type": "Organization", "object_aliases": [], "evidence": "battery pack is built by Voltix"}},
    {{"subject": "Voltix", "subject_type": "Organization", "subject_aliases": [], "relation": "LOCATED_IN",
      "object": "Lyon", "object_type": "Place", "object_aliases": [], "evidence": "built by Voltix in Lyon"}}]}}

Reply with JSON only, no markdown fence:
{{"triples": [{{"subject": "...", "subject_type": "...", "subject_aliases": [], "relation": "...",
  "object": "...", "object_type": "...", "object_aliases": [], "evidence": "..."}}]}}

TEXT (from {chunk['source']}):
{chunk['text']}"""


def _parse(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return Extraction.model_validate(json.loads(raw))


def _valid(t, profile=SUPPORT):
    ok = t.subject_type in profile["types"] and t.object_type in profile["types"] and t.relation in profile["relations"] \
        and t.subject.strip() and t.object.strip() and t.subject.strip().lower() != t.object.strip().lower() \
        and (t.relation != "REPLACES" or t.subject_type == t.object_type)
    allowed = profile["rel_types"].get(t.relation)
    return bool(ok and (allowed is None or (t.subject_type in allowed[0] and t.object_type in allowed[1])))


def extract_chunk(chunk, profile=SUPPORT):
    """One LLM call (a second one if the JSON is invalid). Returns (kept triples as dicts, dropped count).
    The chunk goes to the provider with its personal data replaced; the triples get the real values back.
    Raises llm.LLMError when the model cannot be reached or answers twice with invalid JSON."""
    red = pii.Redactor()
    safe = {**chunk, "text": red.redact(chunk["text"])}
    last = None
    for _ in range(2):
        raw = llm._complete(_prompt(safe, profile))
        try:
            ex = _parse(raw)
            break
        except (json.JSONDecodeError, ValidationError) as e:
            last = e
    else:
        raise llm.LLMError(f"invalid JSON twice for {chunk['id']}: {last}")
    canon_t = {x.lower(): x for x in profile["types"]}
    for t in ex.triples:  # tolerate "owned_by" / "service" written in another case
        t.relation = t.relation.strip().upper().replace(" ", "_")
        t.subject_type = canon_t.get(t.subject_type.strip().lower(), t.subject_type)
        t.object_type = canon_t.get(t.object_type.strip().lower(), t.object_type)
        t.subject, t.object, t.evidence = red.restore(t.subject), red.restore(t.object), red.restore(t.evidence)
        t.subject_aliases = [red.restore(a) for a in t.subject_aliases]
        t.object_aliases = [red.restore(a) for a in t.object_aliases]
    kept = [t.model_dump() for t in ex.triples if _valid(t, profile)]
    return kept, len(ex.triples) - len(kept)


# --------------------------------------------------------------------- entity resolution

def norm_name(name):
    n = re.sub(r"[`*_\"']", "", name.lower()).strip()
    n = re.sub(r"^(the|a|an)\s+", "", n)
    n = re.sub(r"\s+(service|team|plan|policy|channel)$", "", n)
    return re.sub(r"\s+", " ", n).strip()


def build_graph(extractions):
    """extractions: {chunk_id: [triple dicts]}. Returns {'nodes': {...}, 'edges': [...]} with provenance."""
    nodes, alias_to_key, edges = {}, {}, {}

    def node(name, typ, aliases, chunk_id):
        keys = {norm_name(name)} | {norm_name(a) for a in aliases if a.strip()}
        found = next((alias_to_key[(typ, k)] for k in keys if (typ, k) in alias_to_key), None)
        key = found or (typ, norm_name(name))
        n = nodes.setdefault(key, {"name": name.strip(), "type": typ, "names": set(), "chunks": set()})
        n["names"] |= {name.strip()} | {a.strip() for a in aliases if a.strip()}
        n["chunks"].add(chunk_id)
        for k in keys:
            alias_to_key.setdefault((typ, k), key)
        return key

    for chunk_id, triples in extractions.items():
        for t in triples:
            s = node(t["subject"], t["subject_type"], t.get("subject_aliases", []), chunk_id)
            o = node(t["object"], t["object_type"], t.get("object_aliases", []), chunk_id)
            e = edges.setdefault((s, t["relation"], o), {"src": s, "rel": t["relation"], "dst": o, "evidence": {}})
            e["evidence"].setdefault(chunk_id, t["evidence"])
    return {"nodes": nodes, "edges": list(edges.values())}


def label(graph, key):
    return graph["nodes"][key]["name"]


# --------------------------------------------------------------------- build (cached on disk)

def _path(name, profile=SUPPORT):
    # the support profile keeps the V3 file name ("<index>__v3.json"), so the graph already built is still found
    tag = f"v{profile['version']}" if profile["name"] == "support" else f"{profile['name']}_v{profile['version']}"
    return os.path.join(GRAPH_DIR, f"{name}__{tag}.json")


def _empty_cache():
    return {"extractions": {}, "failed": {}, "cost": {"calls": 0, "tokens": 0, "seconds": 0.0, "dropped_triples": 0}}


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_cache(name, profile=SUPPORT):
    return _read_json(_path(name, profile), _empty_cache())


def save_cache(name, cache, profile=SUPPORT):
    _write_json(_path(name, profile), cache)


# One extra store, shared by every index of the same profile: chunk text (hash) -> its triples. Upload a 6th document
# and the 5 documents already read do not cost a single LLM call again.
def _shared_path(profile):
    return os.path.join(GRAPH_DIR, f"_by_text_{profile['name']}_v{profile['version']}.json")


def _text_hash(chunk):
    return hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()


def build_cache(name, chunks, pause=0.0, log=print, profile=SUPPORT, progress=None):
    """Extract every chunk that is not yet in the cache. Safe to stop and restart.
    progress(done, total, chunk_id) is called after each chunk (the UI shows a progress bar with it)."""
    cache = load_cache(name, profile)
    shared = _read_json(_shared_path(profile), {})
    todo = [c for c in chunks if c["id"] not in cache["extractions"]]
    for i, c in enumerate(todo, start=1):
        calls0, tokens0, t0 = llm.STATS["calls"], llm.STATS["tokens"], time.perf_counter()
        with tracing.span("graph.chunk", chunk=c["id"]) as sp:
            h = _text_hash(c)
            if h in shared:  # same text already read (another document set): reuse, no LLM call
                cache["extractions"][c["id"]] = shared[h]
                cache["failed"].pop(c["id"], None)
                sp.set(cache_hit=True, triples=len(shared[h]))
            else:
                try:
                    triples, dropped = extract_chunk(c, profile)
                    cache["extractions"][c["id"]] = triples
                    shared[h] = triples
                    cache["failed"].pop(c["id"], None)
                    cache["cost"]["dropped_triples"] += dropped
                    used = llm.STATS["calls"] - calls0
                    sp.set(cache_hit=False, triples=len(triples), dropped=dropped, llm_calls=used, retried=used > 1)
                    if not triples:
                        sp.set(note="no triples")  # quiet failure: a chunk with no triples raises no error
                except llm.LLMError as e:
                    cache["failed"][c["id"]] = str(e)
                    sp.error(e)
        cache["cost"]["calls"] += llm.STATS["calls"] - calls0
        cache["cost"]["tokens"] += llm.STATS["tokens"] - tokens0
        cache["cost"]["seconds"] = round(cache["cost"]["seconds"] + time.perf_counter() - t0, 2)
        if log:
            log(f"  [{i}/{len(todo)}] {c['id']}: " + (f"{len(cache['extractions'][c['id']])} triples"
                if c["id"] in cache["extractions"] else "FAILED (will be retried on the next build)"))
        if progress:
            progress(i, len(todo), c["id"])
        if i % 5 == 0:
            save_cache(name, cache, profile)
            _write_json(_shared_path(profile), shared)
        if pause:
            time.sleep(pause)
    save_cache(name, cache, profile)
    _write_json(_shared_path(profile), shared)
    return cache


_graphs = {}


def get_graph(name, chunks, build=False, profile=SUPPORT):
    """The resolved graph for these chunks. build=False: use only what is in the cache (never calls the LLM)."""
    cache = build_cache(name, chunks, profile=profile) if build else load_cache(name, profile)
    ids = {c["id"] for c in chunks}
    ex = {cid: t for cid, t in cache["extractions"].items() if cid in ids}
    sig = (name, profile["name"], tuple(sorted((cid, len(t)) for cid, t in ex.items())))
    if sig not in _graphs:
        g = build_graph(ex)
        g["cost"], g["failed"], g["chunks_done"], g["chunks_total"] = cache["cost"], cache["failed"], len(ex), len(ids)
        _graphs[sig] = g
    return _graphs[sig]


def stats(graph):
    kinds = defaultdict(int)
    for n in graph["nodes"].values():
        kinds[n["type"]] += 1
    rels = defaultdict(int)
    for e in graph["edges"]:
        rels[e["rel"]] += 1
    return {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "nodes_by_type": dict(kinds),
            "edges_by_relation": dict(rels), "chunks_done": graph["chunks_done"], "chunks_total": graph["chunks_total"],
            "build_cost": graph["cost"], "failed_chunks": graph["failed"]}


# --------------------------------------------------------------------- the concept map explorer (V4, part B)

def node_id(key):
    return f"{key[0]}|{key[1]}"


def _key(graph, nid):
    return next((k for k in graph["nodes"] if node_id(k) == nid), None)


def explore(graph, limit=150):
    """Nodes and edges for the map. At most `limit` nodes (the best connected ones) so that the page stays readable."""
    degree = defaultdict(int)
    for e in graph["edges"]:
        degree[e["src"]] += 1
        degree[e["dst"]] += 1
    keys = sorted(graph["nodes"], key=lambda k: (-degree[k], graph["nodes"][k]["name"]))[:limit]
    kept = set(keys)
    return {"nodes": [{"id": node_id(k), "name": graph["nodes"][k]["name"], "type": graph["nodes"][k]["type"],
                       "degree": degree[k], "chunks": len(graph["nodes"][k]["chunks"])} for k in keys],
            "edges": [{"src": node_id(e["src"]), "dst": node_id(e["dst"]), "rel": e["rel"], "chunks": len(e["evidence"])}
                      for e in graph["edges"] if e["src"] in kept and e["dst"] in kept],
            "total_nodes": len(graph["nodes"]), "shown_nodes": len(keys)}


def node_detail(graph, nid, chunks_by_id):
    """Everything about one concept: its links (with the quote and the chunk each one was extracted from),
    the chunks that mention it, and the documents they belong to."""
    key = _key(graph, nid)
    if key is None:
        return None
    n = graph["nodes"][key]
    links = []
    for e in graph["edges"]:
        if key not in (e["src"], e["dst"]):
            continue
        out = e["src"] == key
        other = e["dst"] if out else e["src"]
        links.append({"direction": "out" if out else "in", "relation": e["rel"],
                      "other": {"id": node_id(other), "name": label(graph, other), "type": graph["nodes"][other]["type"]},
                      "evidence": [{"chunk_id": cid, "quote": q} for cid, q in sorted(e["evidence"].items())]})
    chunks = [{"id": cid, "source": chunks_by_id[cid]["source"], "chunk_index": chunks_by_id[cid]["chunk_index"],
               "text": chunks_by_id[cid]["text"]} for cid in sorted(n["chunks"]) if cid in chunks_by_id]
    return {"id": nid, "name": n["name"], "type": n["type"], "also_called": sorted(n["names"] - {n["name"]}),
            "links": sorted(links, key=lambda x: (x["relation"], x["other"]["name"])), "chunks": chunks,
            "documents": sorted({c["source"] for c in chunks})}


# --------------------------------------------------------------------- retrieval (no LLM)

def _words(text):
    return " " + re.sub(r"[^a-z0-9\-]+", " ", text.lower()).strip() + " "


def find_seeds(graph, question):
    """Nodes whose name (or alias) appears in the question, matched on word boundaries."""
    q = _words(question)
    squashed = re.sub(r"[^a-z0-9]", "", question.lower())
    seeds = []
    for key, n in graph["nodes"].items():
        for nm in n["names"] | {n["name"]}:
            w = _words(norm_name(nm)).strip()
            if len(w) < 3:
                continue
            if f" {w} " in q or (any(ch.isdigit() for ch in w) and re.sub(r"[^a-z0-9]", "", w) in squashed):
                seeds.append(key)
                break
    return seeds


def walk(graph, seeds, hops=MAX_ASSIGNED_HOPS):
    """Breadth-first walk (edges in both directions). Returns {edge index: distance} (1 = edge touches a seed)."""
    adj = defaultdict(list)
    for i, e in enumerate(graph["edges"]):
        adj[e["src"]].append((i, e["dst"]))
        adj[e["dst"]].append((i, e["src"]))
    dist, seen = {}, {s: 0 for s in seeds}
    queue = deque(seeds)
    while queue:
        cur = queue.popleft()
        if seen[cur] >= hops:
            continue
        for i, nxt in adj[cur]:
            dist[i] = min(dist.get(i, 99), seen[cur] + 1)
            if nxt not in seen:
                seen[nxt] = seen[cur] + 1
                queue.append(nxt)
    return dist


def retrieve(graph, chunks_by_id, question, n=10, hops=MAX_ASSIGNED_HOPS):
    """Chunks ranked by graph evidence: a seed's own chunks count 1, an edge at distance d adds 1/(d+1) to each chunk
    that states it. Each returned chunk carries the human-readable 'path' (the edges that led to it)."""
    seeds = find_seeds(graph, question)
    score, paths = defaultdict(float), defaultdict(list)
    for s in seeds:
        for cid in graph["nodes"][s]["chunks"]:
            score[cid] += 1.0
            paths[cid].append(f"mentions {label(graph, s)}")
    for i, d in walk(graph, seeds, hops).items():
        e = graph["edges"][i]
        text = f"{label(graph, e['src'])} -{e['rel']}-> {label(graph, e['dst'])}"
        for cid in e["evidence"]:
            score[cid] += 1.0 / (d + 1)
            paths[cid].append(f"{text} (hop {d})")
    ranked = sorted((c for c in score if c in chunks_by_id), key=lambda c: (-score[c], c))[:n]
    return [{**chunks_by_id[cid], "rank": r, "graph_score": round(score[cid], 3), "path": paths[cid]}
            for r, cid in enumerate(ranked, start=1)], [label(graph, s) for s in seeds]


# --------------------------------------------------------------------- hand grading of the extraction

def sample_triples(graph_cache, chunks, n=10, seed=7):
    by_id = {c["id"]: c for c in chunks}
    pool = [(cid, t) for cid, ts in sorted(graph_cache["extractions"].items()) for t in ts]
    random.Random(seed).shuffle(pool)
    return [{"chunk_id": cid, "chunk_text": by_id[cid]["text"] if cid in by_id else "",
             "triple": f"{t['subject']} ({t['subject_type']}) -{t['relation']}-> {t['object']} ({t['object_type']})",
             "evidence": t["evidence"], "grade": None, "note": ""} for cid, t in pool[:n]]


def count_grades(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    n = {"correct": 0, "wrong": 0, "unsupported": 0, "ungraded": 0}
    for r in rows:
        n[r["grade"] if r["grade"] in n else "ungraded"] += 1
    return n


# --------------------------------------------------------------------- command line

def _setup(corpus, cs, ov):
    import rag
    chunks = rag.build_chunks(corpus, cs, ov)
    return chunks, rag._index_name(corpus, cs, ov), profile_for(corpus)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "stats", "sample", "grades", "ask", "debug"])
    ap.add_argument("target", help="corpus name (grades: path of the triples file)")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between calls (rate limit)")
    a = ap.parse_args(argv)
    if a.command == "grades":
        print(count_grades(a.target))
        return
    chunks, name, profile = _setup(a.target, a.chunk_size, a.overlap)
    if a.command == "build":
        with tracing.trace("index", corpus=a.target, via="command line", config={"chunk_size": a.chunk_size,
                           "chunk_overlap": a.overlap, "graph_profile": profile["name"]}):
            with tracing.span("graph.extract", profile=profile["name"], chunks=len(chunks)):
                c = build_cache(name, chunks, pause=a.pause, profile=profile)
        print("\nbuild cost so far:", c["cost"], "| failed chunks:", len(c["failed"]))
    elif a.command == "stats":
        print(json.dumps(stats(get_graph(name, chunks, profile=profile)), indent=1))
    elif a.command == "debug":  # python graph.py debug kestrel "service-catalog.md::2"
        chunk = next((c for c in chunks if c["id"] == a.question), None)
        if not chunk:
            sys.exit(f"No chunk {a.question!r}. Example ids: {[c['id'] for c in chunks][:5]}")
        print("RAW MODEL ANSWER:\n", llm._complete(_prompt(chunk, profile)))
    elif a.command == "sample":
        rows = sample_triples(load_cache(name, profile), chunks)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, f"graph_triples_{a.target}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"Saved {out}. Fill each \"grade\" with correct / wrong / unsupported, then: python graph.py grades {out}")
    elif a.command == "ask":
        g = get_graph(name, chunks, profile=profile)
        got, seeds = retrieve(g, {c["id"]: c for c in chunks}, a.question or "")
        print("seed entities:", seeds)
        for c in got:
            print(f"{c['rank']:>2} {c['id']} score={c['graph_score']}")
            for p in c["path"]:
                print("     ", p)


if __name__ == "__main__":
    main(sys.argv[1:])
