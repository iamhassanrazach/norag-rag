# REPORT - RAG project

<!-- SKELETON. The course wants this report in YOUR words, not AI-generated: fill every section yourself. -->

## Team
<!-- names, GitHub accounts, who did what -->

## Results table (V2)
<!-- Paste YOUR numbers from results/*.json. One table per corpus: rows = configuration x question type,
     columns = hit@1/3/5/10 and recall@1/3/5/10. -->

| Corpus | Config | Type | n | hit@1 | hit@3 | hit@5 | hit@10 | rec@1 | rec@3 | rec@5 | rec@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|

## What I changed between configurations, and what moved
<!-- Which setting(s) changed, which types went up or down, and your explanation. -->

## Two diagnosed misses (Words tab, question mode)
### Kestrel: question ___
<!-- Do the question's words appear in the chunk holding the answer? In many other chunks? Absent altogether? -->
### Own corpus: question ___

## No-context test
<!-- Which questions did the model answer correctly without retrieval, in each set? Replaced or kept (and why)? -->

## Answer grading
<!-- Grading rules (written BEFORE grading), chosen question set, grades, retrieval vs generation failures. -->

## Decision log
<!-- Each decision + the reason. Decisions already made in the code, to confirm or change in your own words:
     - relevance = right source file + quote contained in the chunk (quote cut by a boundary can never match)
     - HitRate@k and Recall@k, unanswerable questions excluded
     - words: lowercase, hyphenated identifiers kept whole (KST-2044), no stemming, tiny stop list
     - overlapping chunks: a word in the overlap counts in both chunks
     - index name includes a hash of the documents' content
     - answers: temperature 0, JSON validated with Pydantic, "Not in the documents." when in_documents is false -->

## Known failures
