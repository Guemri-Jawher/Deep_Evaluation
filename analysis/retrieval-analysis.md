# Retrieval evaluation report

Runs analysed (files in `reports/`): all cases `20261025T2111*`, `--split dev` `20261025T212406/7`, `--split heldout` `20261025T212448/9`.
Dataset `datasets/data/v1/rag_golden.jsonl`, k=10, only answerable questions (49 = 41 dev + 8 heldout; the 5 `unanswerable` rows have no evidence and are skipped).

## 1. How to read the metrics

A retrieved chunk is "relevant" if it is on a gold page of the gold source and contains a gold phrase. Labels were validated against the corpus: every evidence unit matches at least one chunk, so no failure below comes from an unreachable label.

| Metric | Meaning | Good when |
|---|---|---|
| `recall@5` | share of the question's evidence units found in the top 5 | high; the main one. If the answer is not in the top 5, prompt tuning cannot fix it. |
| `recall@1`, `recall@10` | same with a stricter or looser cut | `@10` far above `@5` means the evidence is found but ranked too low |
| `mrr` | 1 / rank of the first relevant chunk, averaged | high; 1.0 means always first |
| `ndcg@5` | rewards relevant chunks that appear higher | high |
| `precision@5` | share of the top 5 that is relevant | do not judge on it: most questions have only 1-2 relevant chunks, so it cannot go much above 0.2-0.4 |
| `pass_rate` | share of questions with `recall@5 = 1` | high; differs from `recall@5` only on multi-evidence questions |

The `[low, high]` range is a 95% bootstrap interval: the plausible range of the true score. Two strategies whose ranges overlap cannot be told apart.

## 2. How to judge: the three runs, in order

| Run | Question it answers | How to use it |
|---|---|---|
| all (`run_retrieval_eval.py`) | overall picture | first look, nothing is decided here |
| `--split dev` | what fails and why | iterate here: change chunk size, `k`, strategy, then rerun |
| `--split heldout` | did the improvement hold on questions I never tuned on | run once at the end; never tune on its failures |

Compare dev and heldout. A system that is much better on dev than on heldout is overfit to dev.

## 3. Results

Mean (95% interval in the all-cases run).

| Run | n | strategy | recall@1 | recall@5 | recall@10 | mrr | ndcg@5 |
|---|---|---|---|---|---|---|---|
| all | 49 | vector | 0.633 | 0.898 [0.82, 0.98] | 0.949 | 0.754 | 0.758 |
| all | 49 | bm25 | 0.673 | 0.898 [0.80, 0.98] | 0.939 | 0.773 | 0.757 |
| all | 49 | hybrid | 0.684 | 0.888 [0.80, 0.97] | 0.939 | **0.798** | **0.778** |
| dev | 41 | vector | 0.610 | 0.878 [0.78, 0.95] | 0.939 | 0.734 | 0.737 |
| dev | 41 | bm25 | 0.634 | 0.878 [0.78, 0.98] | 0.927 | 0.748 | 0.729 |
| dev | 41 | hybrid | 0.646 | 0.866 [0.76, 0.95] | 0.927 | **0.774** | **0.751** |
| heldout | 8 | vector | 0.750 | 1.000 | 1.000 | 0.854 | 0.867 |
| heldout | 8 | bm25 | 0.875 | 1.000 | 1.000 | 0.900 | 0.905 |
| heldout | 8 | hybrid | 0.875 | 1.000 | 1.000 | **0.917** | **0.913** |

What is good:
- About 90% of questions have their evidence in the top 5, and about 94-95% in the top 10.
- 28 of 49 questions are perfect on all three strategies (the first result is relevant).
- Heldout is not worse than dev, so there is no sign of overfitting to dev.

What is not good or not proven:
- **No strategy clearly wins.** `recall@5` differs by about 0.01 between strategies while each interval is about 0.15-0.2 wide. Hybrid has the best `mrr` and `ndcg@5` (relevant chunks ranked higher), but the gap is inside the intervals.
- **Heldout `recall@5 = 1.000 [1.00, 1.00]` is not a proof of perfection.** With 8 questions and no failure the bootstrap has nothing to resample, so the interval collapses. Read it as "no failure on 8 questions".
- **Ranking is weaker than recall:** `recall@1` is only 0.63-0.68, so the first result is relevant for about two questions in three.
- 3 multi-hop questions and 4 numeric questions are too few to compare strategies by category.

## 4. Recall@5 by category (all cases)

| category | n | vector | bm25 | hybrid |
|---|---|---|---|---|
| factual | 23 | 0.96 | 0.91 | 0.91 |
| explanation | 12 | 1.00 | 0.92 | 0.92 |
| procedure | 6 | 0.67 | 0.83 | 0.83 |
| numeric | 4 | 0.75 | 0.75 | 0.75 |
| multi_hop | 3 | 0.67 | 1.00 | 0.83 |
| diagnosis | 1 | 1.00 | 1.00 | 1.00 |

Procedure and numeric questions are the weak spots. Vector is best on explanation and factual questions, BM25 on procedure and multi_hop, which is the argument for hybrid.

## 5. Every tested case

### 5.1 Perfect on all three strategies (28 cases, first result relevant)

r004 r005 r006 r007 r008 r009 r010 r011 r012 r013 r015 r016 r018 r020 r024 (dev), r025 r026 r027 r028 (heldout), r030 r036 r039 r041 r042 (dev), r043 r044 (heldout), r047 r048 (dev, multi_hop: both guides found).

### 5.2 Found but not always first (21 cases)

The values are the rank of the first relevant chunk (1 is best, `-` means not in the top 10). `✗` marks `recall@5 < 1`, i.e. a failed case.

| case | category | split | vector | bm25 | hybrid | note |
|---|---|---|---|---|---|---|
| r001 | factual | dev | - ✗ | 1 | 6 ✗ | BM25 finds it, vector misses it, fusion dilutes it |
| r002 | factual | dev | 5 | 1 | 1 | |
| r003 | factual | dev | 1 | 2 | 1 | |
| r014 | factual | dev | 2 | 1 | 2 | |
| r017 | explanation | dev | 1 | 4 | 2 | |
| r019 | factual | dev | 4 | 4 | 4 | low on all |
| r021 | explanation | dev | 2 | 1 | 1 | |
| r022 | factual | dev | 3 | 2 | 2 | |
| r023 | factual | dev | 2 | 4 | 2 | |
| r029 | factual | dev | 1 | - ✗ | - ✗ | vector finds it, BM25 and hybrid miss it |
| r031 | procedure | dev | 7 ✗ | 3 | 3 | vector too low |
| r032 | explanation | dev | 5 | - ✗ | - ✗ | vector finds it, BM25 and hybrid miss it |
| r033 | explanation | dev | 2 | 1 | 1 | |
| r034 | procedure | dev | - ✗ | - ✗ | - ✗ | **missed by all, not even in the top 10** |
| r035 | numeric | dev | 9 ✗ | 7 ✗ | 6 ✗ | **failed by all, found between ranks 6 and 10** |
| r037 | procedure | dev | 4 | 2 | 2 | |
| r038 | factual | dev | 2 | 10 ✗ | 3 | BM25 too low |
| r040 | explanation | dev | 1 | 2 | 1 | |
| r045 | factual | heldout | 3 | 5 | 3 | |
| r046 | procedure | heldout | 2 | 1 | 1 | |
| r049 | multi_hop | dev | 0 of 2 in top 5 ✗ | 2 of 2 | 1 of 2 ✗ | needs two evidence units; only BM25 gets both in the top 5, and hybrid finds both within the top 10 |

### 5.3 Failures (`recall@5 < 1`)

| strategy | n | cases |
|---|---|---|
| vector | 5 | r001, r031, r034, r035, r049 |
| bm25 | 5 | r029, r032, r034, r035, r038 |
| hybrid | 6 | r001, r029, r032, r034, r035, r049 |

- **Failed by every strategy:** r034 (reflection loop) and r035 (router confidence). Fix these first. The labels are valid (r034 expects `agentic-patterns:p14:c2` and `p15:c1`, r035 expects `agentic-patterns:p17:c0`). For r035 the evidence is at ranks 6-10, so a larger `k` or a better ranking would fix it.
- **Failed by one retriever only:** r001 and r031 (vector), r029, r032 and r038 (BM25). Each is found by the other retriever. Hybrid does not always recover them: for r029 and r032 hybrid inherits the BM25 miss, and r001 is pushed from vector's miss to rank 6.
- Likely causes, **not verified**: BM25 depends on shared words, so it fails when the question is worded differently from the guide (r029, r032). Embeddings handle numbers and exact terms poorly (r035). A phrase may be split across chunks (r034).

## 6. Verdict

Retrieval is usable: about 90% `recall@5`, stable between dev and heldout, with 28 of 49 questions perfect. It is not yet something to ship on faith, because:
1. no strategy is proven better than another (intervals overlap),
2. heldout has only 8 questions,
3. two questions (r034, r035) fail on every strategy.

## 7. Next steps

1. Look at the chunks around the relevant ones for r034 and r035, and the failing questions' wording against the guide text.
2. Try changes on **dev only** (`chunk_size`, `chunk_overlap`, pool size in hybrid, `k`), and compare runs with `compare_reports` so that a gain is judged by its interval, not by the mean.
3. Run `--split heldout` once to confirm.
4. Retrieval does not test refusal. The 5 `unanswerable` rows need the full pipeline and an LLM (`The guides do not cover this.`); that evaluation is not done yet.


# Example
Example with a multi-hop question that has 2 evidence units, k = 5, and top 5 [A, X, B, Y, Z], where A matches unit 1, B matches unit 2, and X, Y, Z are irrelevant:

recall@5 = 2/2 = 1.0, because both units are found.
precision@5 = 2/5 = 0.4, because 3 of the 5 chunks are noise.
For a single-unit question with one relevant chunk in the top 5, recall@5 = 1.0 and precision@5 = 0.2. This is why precision@5 is about 0.21 in your reports even when retrieval works. Most questions have only 1-2 relevant chunks, so precision has a low ceiling. Use recall to judge your retriever, since a missing chunk means the LLM never sees the answer.

reciprocal_rank mean
It measures ranking, not only presence. recall@5 gives the same score (1.0) whether the right chunk is at rank 1 or rank 5. reciprocal_rank tells them apart. Ranking matters in RAG because the LLM reads the top chunks first, and a relevant chunk at rank 1 is more likely to be used than one buried at rank 5.