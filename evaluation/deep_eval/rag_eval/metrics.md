# DeepEval metrics used in the RAG tests

Every metric here is **LLM-judged**: a judge model reads the question, the retrieved chunks and the answer, and returns verdicts that DeepEval turns into a score between 0 and 1. Metrics are defined in `case.py` and run by `suite.py`. Scores below were checked against the DeepEval 4.3.1 source.

## 1. Which metric runs where

| Metric | Report key | Retrieval suite | RAG suite (answerable) | RAG suite (unanswerable) |
|---|---|---|---|---|
| Contextual relevancy | `de_contextual_relevancy` | yes | yes | no |
| Contextual precision | `de_contextual_precision` | yes | yes | no |
| Contextual recall | `de_contextual_recall` | yes | yes | no |
| Faithfulness | `de_faithfulness` | no | yes | yes |
| Answer relevancy | `de_answer_relevancy` | no | yes | yes |
| Correctness (`GEval`) | `de_correctness` | no | yes | no |

- **Retrieval suite** (`run_deepeval_retrieval_suite`): retrieves the top `k` chunks (default 5) and judges only the context. No LLM answer exists, so the reference answer is used as `actual_output`.
- **RAG suite** (`run_deepeval_rag_suite`): runs the real pipeline and judges the context and the answer.

## 2. How a score becomes pass or fail

- A metric **passes** when `score >= THRESHOLD` (`THRESHOLD = 0.7` in `case.py`, used by every metric).
- A **case passes** only if **all** its metrics pass. One low metric fails the case.
- A case that **crashes** (judge error, rate limit, bad JSON) is a FAIL with no scores. Check `error_rate` before reading anything else.
- Scores are fractions, so they come in coarse steps. With 5 chunks or 3 statements, a score can only be 0, 0.33, 0.67, 1 and so on.

## 3. The metrics

### Contextual relevancy
- **Question:** is what was retrieved relevant to the question, or is it noise?
- **Needs:** question, retrieved chunks. No reference answer.
- **How:** the judge splits each chunk into statements and marks each as relevant or not to the question.
- **Score:** relevant statements / all statements. 0 if there are none.
- **High (1.0):** everything retrieved is on topic. **Low:** the context is full of unrelated text.
- **Calls:** one per chunk, plus one for the reason (5 chunks gave 5 + 1 calls in an observed run).

### Contextual precision
- **Question:** are the relevant chunks ranked **above** the irrelevant ones?
- **Needs:** question, retrieved chunks, reference answer.
- **How:** each chunk, in rank order, is marked relevant or not. The score averages the precision at each relevant position.
- **Score:** mean of `precision@k` over the relevant ranks; 0 if no chunk is relevant. Order matters:

| Relevance by rank | Score |
|---|---|
| relevant, no, no, no, no | 1.00 |
| relevant, no, relevant, no, no | 0.83 |
| no, no, no, no, relevant | 0.20 |
| none relevant | 0.00 |

- **Low with a relevant chunk present:** the right chunk is retrieved but ranked too low.

### Contextual recall
- **Question:** does the retrieved context contain what is needed to write the reference answer?
- **Needs:** question, retrieved chunks, reference answer.
- **How:** the reference answer is split into sentences. The judge checks whether each sentence can be attributed to the context.
- **Score:** attributed sentences / all reference sentences. 0 if there are none.
- **Low:** key facts of the answer are missing from the retrieved chunks. This is the closest DeepEval has to recall.

### Faithfulness
- **Question:** does the answer say only what the context supports (no hallucination)?
- **Needs:** question, the answer, retrieved chunks.
- **How:** the judge extracts facts from the context and claims from the answer, then marks each claim yes, no or unsure.
- **Score:** (claims marked yes or unsure) / all claims. **1.0 if the answer makes no claims.**
- **Consequence:** an abstention ("The guides do not cover this.") scores 1.0, which is why it is the metric for unanswerable questions. An unsure verdict counts as a pass, so the metric is lenient.

### Answer relevancy
- **Question:** does the answer address the question asked?
- **Needs:** question, the answer. It does not use the context or the reference.
- **How:** the judge extracts statements from the answer and marks each as relevant to the question (yes, unsure or no).
- **Score:** (yes or unsure) / all statements. 1.0 if there are none.
- **Low:** the answer is off-topic or padded.
- **Not a correctness check:** a fluent wrong answer on topic can score 1.0.
- **Unanswerable cases:** it is also run there. I have not tested how it scores an abstention.

### Correctness (custom `GEval`)
- **Question:** is the answer factually consistent with the reference answer?
- **Needs:** question, the answer, reference answer.
- **How:** DeepEval generates evaluation steps from our criteria (penalize missing key facts and contradictions, ignore wording and citations), then the judge gives an integer score from 0 to 10.
- **Score:** `score / 10`, so 0.0 to 1.0. The values are probably coarse (multiples of 0.1), since our judge adapter does not return token probabilities for weighting.
- **Low:** the answer misses facts or contradicts the reference. It is the only metric that compares the answer to the gold answer.

## 4. The differences that matter

**Retrieval or generation**

| Group | Metrics | What a low score points to |
|---|---|---|
| Retrieval | relevancy, precision, recall | the retriever or the chunking |
| Generation | faithfulness, answer relevancy, correctness | the prompt or the LLM |

**Reference needed or not**
- **Reference-free** (judge sees only the question, context and answer): contextual relevancy, faithfulness, answer relevancy.
- **Reference-based** (also needs the gold answer): contextual precision, contextual recall, correctness.

**The RAG triad** (from the evaluation guide)

| Triad part | Metric |
|---|---|
| context relevance | contextual relevancy |
| groundedness | faithfulness |
| answer relevance | answer relevancy |

**Reading failures by pattern**

| Pattern | Likely cause |
|---|---|
| recall low | the answer was never in the retrieved chunks |
| recall fine, precision low | the right chunk is retrieved but ranked low |
| relevancy low | too much irrelevant text in the context |
| context good, faithfulness low | the model ignores or contradicts the context |
| faithfulness high, correctness low | the model abstained, or answered from the context but missed key facts |
| answer relevancy low | off-topic or evasive answer |

Example: r001 scored **0.00 on all three contextual metrics**. The judge reasons say none of the five chunks contained the four loop steps. That agrees with the label-based report, where hybrid retrieval misses r001 in the top 5.

## 5. Metrics derived by the runner

`summarize` in `evaluation/core/runner.py` adds these to every report:

| Metric | Meaning |
|---|---|
| `de_<metric>` | mean score over the cases, with a 95% bootstrap interval and `n` |
| `pass_rate` | share of cases where all metrics passed |
| `error_rate` | share of cases that crashed |
| `latency_p50_s`, `latency_p95_s` | median and slow-tail time per case |
| `tokens_per_case` | token use, if reported |

## 6. DeepEval metrics vs the hand-written ones

DeepEval has no `recall@k`, `hit@k`, `mrr` or `ndcg`. Its recall and precision are LLM judgments about content, not checks against gold pages.

| Hand-written (`hard_coded_lib`) | Closest DeepEval metric | Difference |
|---|---|---|
| `recall@k` | contextual recall | gold evidence found vs. the reference answer supported by the context |
| `precision@k` | contextual relevancy | gold phrases vs. the judge's view of relevance |
| `mrr`, `ndcg@5` | contextual precision | first or discounted rank of gold chunks vs. rank-weighted relevance by the judge |
| none | faithfulness, answer relevancy, correctness | answer quality is not measured by the hand-written suite |

## 7. Limits to keep in mind

- Scores depend on the judge model and can vary between runs. Check a few cases by hand before trusting them.
- One threshold (0.7) applies to every metric, and a case fails if any single metric is below it.
- `measure()` in `suite.py` turns a missing score into 0.0 (`metric.score or 0.0`), so an unscored metric looks like a real zero.
- Cost is high: 10 judge calls for the three contextual metrics on one case, and more once the RAG metrics are added. The count for the other metrics was not measured.
