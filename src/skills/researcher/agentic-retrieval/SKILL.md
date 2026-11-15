---
name: agentic-retrieval
description: >-
  Query-rewriting and relevance-grading loop for gathering evidence from the guides with
  search_docs: decompose the brief, retrieve, grade each passage, rewrite weak queries, and report
  missing facts. Use for every research brief the supervisor delegates.
license: MIT
metadata:
  version: "1.0"
  owner: kb-agent
---

# Agentic retrieval

## Loop (per fact)

1. Query 1: 3-6 keywords from the brief plus the guide's terminology.
2. Grade each passage: relevant (answers the fact), partial, or noise. Keep relevant and partial.
3. If nothing relevant: Query 2 = synonym or the section title you expect ("recall at k" -> "RAG triad").
4. If still nothing: Query 3 = broader topic, then stop and list the fact under `Missing`.

## Query patterns that work

- Metric names with their symbol: `pass@k pass^k`, `recall@k precision@k MRR`.
- Section titles: "where it breaks", "build exercise", "common failure modes and fixes".
- Source filter: `eval-guide` for evaluation, `agentic-patterns` for agent design.

## Return

Follow the output format in your system prompt. Evidence only, with exact ids.
