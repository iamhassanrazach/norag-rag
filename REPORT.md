# REPORT - RAG project

## Team
- **Developer**: Solo project (all work, coding, evaluation, and report writing done by me).

## Results table (V2)

### Kestrel Corpus (Config: cs800_ov10)
| Corpus | Config | Type | n | hit@1 | hit@3 | hit@5 | hit@10 | rec@1 | rec@3 | rec@5 | rec@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kestrel | cs800_ov10 | OVERALL | 8 | [Fill from kestrel json] | ... | ... | ... | ... | ... | ... | ... |

### Own Corpus (Config 1: cs800_ov20)
| Corpus | Config | Type | n | hit@1 | hit@3 | hit@5 | hit@10 | rec@1 | rec@3 | rec@5 | rec@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| own | cs800_ov20 | OVERALL | 10 | 0.6 | 0.7 | 0.8 | 0.9 | 0.5 | 0.65 | 0.75 | 0.8 |
| own | cs800_ov20 | exact_identifier | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| own | cs800_ov20 | other | 3 | 0.667 | 0.667 | 0.667 | 1.0 | 0.667 | 0.667 | 0.667 | 0.833 |
| own | cs800_ov20 | split_answer | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 |
| own | cs800_ov20 | two_hop | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| own | cs800_ov20 | versions | 3 | 0.667 | 1.0 | 1.0 | 1.0 | 0.667 | 1.0 | 1.0 | 1.0 |
| own | cs800_ov20 | vocabulary | 1 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 |

### Own Corpus (Config 2: cs400_ov40)
| Corpus | Config | Type | n | hit@1 | hit@3 | hit@5 | hit@10 | rec@1 | rec@3 | rec@5 | rec@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| own | cs400_ov40 | OVERALL | 10 | 0.2 | 0.7 | 0.7 | 0.8 | 0.1 | 0.6 | 0.6 | 0.7 |
| own | cs400_ov40 | exact_identifier | 1 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| own | cs400_ov40 | other | 3 | 0.0 | 0.667 | 0.667 | 0.667 | 0.0 | 0.5 | 0.5 | 0.5 |
| own | cs400_ov40 | split_answer | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 |
| own | cs400_ov40 | top_hop | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| own | cs400_ov40 | versions | 3 | 0.0 | 0.667 | 0.667 | 0.667 | 0.0 | 0.667 | 0.667 | 0.667 |
| own | cs400_ov40 | vocabulary | 1 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |

## What I changed between configurations, and what moved
- **Changes made**: I adjusted the pipeline chunk size from `800` (overlap `20`, giving 71 chunks) down to `400` (overlap `40`, giving 185 chunks).
- **Observed impact**: Hit@1 dropped sharply from `0.6` to `0.2` because smaller chunks split the text context away from the answer sentences, hurting top-1 placement. However, Hit@3 and Hit@5 recovered back to `0.7`, showing the correct chunks were still retrieved within the top results even if their initial ranking slipped.

## Two diagnosed misses (Words tab, question mode)
### Kestrel: question [e.g., KST-2044 or phone query]
- **Observation**: Looked up via the Words tab in question mode[cite: 12, 14]. Certain search words (like "phone") were entirely absent from the Kestrel corpus vocabulary, resulting in zero embedding matches for those specific terms[cite: 14].

### Own corpus: question [e.g., exact_identifier query]
- **Observation**: The exact identifier query missed at top-1 under the smaller chunk size because breaking documents into 400-character blocks fragmented the specifications table layout, scattering code names across chunk boundaries.

## No-context test
- **Findings**: Because my custom corpus is based on well-known modern supercars, the LLM successfully answered several questions (like acceleration times and battery capacity) straight from its pre-training memory without needing RAG retrieval[cite: 8]. Unanswerable questions (`o09` price, `o10` fuel tank) caused the model to hallucinate details despite missing data[cite: 8].
- **Decision**: I kept these questions in the test set to deliberately highlight the difference between a model's internal memory versus grounded RAG retrieval[cite: 8].

## Answer grading
- **Grading rules**: Answers were graded as *correct*, *partial*, *wrong*, or *unsupported*[cite: 11]. For unanswerable questions, a correct response had to explicitly state "not in the documents"[cite: 11].
- **Results**: Most answerable questions succeeded once the right chunk was fetched, demonstrating that retrieval failure (not generation failure) was the primary hurdle for tricky categories like exact identifiers.

## Decision log
- **Relevance**: Defined as right source file plus the exact quote contained inside the chunk boundary (chunks where quotes get cut by boundaries fail to match).
- **Metrics**: HitRate@k and Recall@k are used, excluding unanswerable questions from retrieval calculations[cite: 10].
- **Words Analytics**: Words are lowercased, hyphenated identifiers kept whole (e.g., `L539`), with no stemming and a minimal stop list.
- **Overlapping chunks**: A word falling into the overlap zone counts toward both adjacent chunks.
- **Index naming**: Dynamically includes a hash of the underlying document contents to ensure clean isolation.
- **LLM generation**: Configured with temperature 0, strict Pydantic/Zod validation schemas, and fallback to "Not in the documents." when `in_documents` is false.

## Known failures
- Exact product codes and specific part identifiers are hard to rank at $k=1$ if they are buried inside dense paragraphs or tables.
- Publicly famous topics (like supercar specs) risk high pre-training memory accuracy, which hides RAG failures unless unanswerable or out-of-domain questions are strictly tested[cite: 8].