# REPORT - RAG project

## Team
Hassan, GitHub: `iamhassanrazach`. I am the only one in my team and I did all the work.

## Results table
Final runs, 800-character chunks, 20% overlap, 10 scored questions per corpus.
(Generate the full per-type table with: `python evaluate.py --table results\kestrel__*.json results\own__*.json`)

| Corpus | Configuration | hit@1 | hit@3 | hit@5 | hit@10 | recall@10 |
|---|---|---|---|---|---|---|
| Kestrel | dense | 0.7 | 0.9 | 0.9 | 1.0 | 0.95 |
| Kestrel | bm25 | 0.7 | 0.8 | 0.8 | 0.8 | 0.675 |
| Kestrel | dense + bm25 | 0.7 | 0.8 | 0.8 | 0.8 | 0.8 |
| Kestrel | dense + bm25 + graph | 0.6 | 0.9 | 0.9 | 0.9 | 0.85 |
| Own (cars) | dense | 0.6 | 0.7 | 0.8 | 0.9 | 0.8 |
| Own (cars) | dense + bm25 | 0.5 | 0.7 | 0.8 | 1.0 | 0.9 |

## Default configuration
I would choose the corpus: kestrel and config: dense as default. It is quickest and the simplest configuration. The gap between car corpus is a single question as there are total of 10 questions. one is wrong. The graph costs 48 gemini calls and covers only part of the kestrel.

## BM25 and the vocabulary question
BM25 is designed to find exact words. It considers phone and phones two different words, and the word phone is not even in the Kestrel documents, that is why it misses the word phone.

## Why dense did better than the hybrid
Dense found the correct chunks, and because BM25 did not rank those chunks highly, the correct chunks did not rise up and instead fell out of the top 10 even though they were the right chunks. This is why dense did better than hybrid.

## The concept graph
It costs upfront (48 Gemini calls, once) and then afterwards its cost is zero. But it is not very efficient and sometimes it takes cut off words and makes them nodes that are of no use and are not present anywhere else in the text. Some important words are not even in nodes, and it only returns the chunks that its nodes came from, so what is left out stays out, like KST-2044, which was not present in the graph at all. For q11 the right chunk was missing from the hybrid top 10 and the graph brought it to rank 2.

Extraction quality (10 random triples graded by hand): 8 correct, 2 wrong, 0 unsupported. File: `results/graph_triples_kestrel.json`.

## Entity resolution
The graph should look at words cut by the chunk boundary and instead of treating the words at the end like whole words should treat them as half and should try to find their right node. Like "Analytics E" should have been merged with "Analytics Engine" as it was the same word but cut off by the chunk boundary. And the graph should understand that two mentions at different places can be the same thing and should be one node (it already does this when the names are written the same).

## No-context test
I would keep the questions and explain what Gemini is doing with inventing the answers and answering from memory rather than from documents. As the answers are not in documents, instead of saying not found it invents an answer.

## Two diagnosed misses (Words tab, question mode)

### Kestrel: q01 ("What does error KST-2044 mean and how do I fix it?")
KST-2044 appears in only one chunk. The words "mean" and "fix" are found nowhere in the corpus. The actual answer was in the next chunk, which did not contain any distinctive words from the question, so BM25 failed to find it and dense found it at rank 3.

### Own corpus: o06 (compression ratio of the original Aventador's engine)
The word "compression" appears in 5 chunks and "ratio" in 10 chunks, and both are in the answer chunk in the V12 file. But the words "lamborghini" and "aventador" are mentioned everywhere and carry almost no weight. The ratio however sits under the "Second generation" heading but does not mention the name Aventador. The file named aventador names the engine and the V12 file holds the numbers, so the question needs two files.

## Decision log
- **Keep the headings.** The base code discarded every paragraph shorter than 20 characters, and that also discarded the headings, so Gemini did not know what a paragraph belonged to and wrote nonsense facts at first. After keeping the headings, BM25's two-hop hit@1 improved from 0.33 to 1.0 (recall@10 dipped a little, from 0.725 to 0.675).
- **Drop wrong-direction triples.** A wrong triple is dropped, not repaired, because a dropped fact is just a missed fact, while repairing a triple is guessing and could be wrong, and we do not want wrong facts.
- **Relevance = right file and the quote inside the chunk.** The right file alone is not enough: a big file has many chunks and only one has the answer. The quote proves that the chunk actually contains the answer. The file also matters because files can have almost the same text but different years or versions. A side effect: if the quote is cut by a chunk boundary, no chunk can ever match it.

## Known failures
- BM25 only matches exact words, so it misses a word that is not in the documents ("phone").
- The graph sometimes turns words cut by a chunk boundary into useless nodes ("Analytics E").
- Important words are missing from the graph (KST-2044 is not in it at all), and the billing and product texts gave no facts.
- In the hybrid, the right chunks that only dense found fell out of the top 10.
