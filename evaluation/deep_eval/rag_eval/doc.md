# DeepEval RAG evaluation

LLM-judged evaluation of the RAG system with DeepEval's native metrics only. It does not use the hand-written metrics in `evaluation/hard_coded_lib/` (which have their own, LLM-free suite, run with `scripts/run_retrieval_eval.py`).

**Status:** the code is written but has not been run yet. Nothing calls it, so there is no script for it.

## Files

| File | Role |
|---|---|
| `case.py` | Builds DeepEval test cases from our objects and chooses the metrics for each case type. |
| `suite.py` | Runs the metrics over the golden set and returns a `SuiteReport`. |
| `../model.py` | `LangChainJudge`: lets DeepEval use any LangChain chat model (here OpenRouter) as the judge. |

### `case.py`

- `THRESHOLD = 0.7`: a metric passes when its score is at least this value. Used by every metric.
- `RAG_METRICS`: name to factory, one fresh metric instance per call (DeepEval metrics keep per-case state). All use `async_mode=False` because `measure()` is synchronous.
- `RETRIEVAL_METRICS`: the three contextual metrics used by the retrieval suite.
- `rag_metric_names(case)`: metrics for the full pipeline. Answerable questions get all of `RAG_METRICS`. Unanswerable ones get `RAG_METRICS_UNANSWERABLE` (`faithfulness`, `answer_relevancy`), since there is no gold evidence to compare with.
- `retrieval_test_case(case, chunks)`: no LLM answer exists at this stage, so the reference answer is used as `actual_output` and only the retrieved context is judged.
- `rag_test_case(case, result)`: uses the real pipeline answer as `actual_output`.

Both builders put the retrieved chunks in `retrieval_context` as `[chunk_id] text`.

### `suite.py`

- `measure(test_case, factories, judge)`: runs each metric and returns `scores` (key `de_<metric>`), `reasons` (the judge's explanation) and `failed` (metrics under the threshold).
- `run_deepeval_retrieval_suite(judge_llm, ...)`: retrieves the top `k` chunks (default 5) for each **answerable** question and judges them. Parameters: `strategy`, `k`, `split`, `limit`, `max_workers`.
- `run_deepeval_rag_suite(judge_llm, ...)`: runs the full pipeline (retrieve, then answer) on **all** questions, including unanswerable ones, and judges the answers. Parameters: `pipeline`, `split`, `limit`, `max_workers`.

A case passes only if none of its metrics failed. Case results are aggregated by `evaluation.core.runner.summarize` into the same `SuiteReport` as the native suites, so reports and comparisons work unchanged. Telemetry is turned off by default.

## Metrics

| Metric | Suite | Judges | Needs |
|---|---|---|---|
| `contextual_relevancy` | retrieval, rag | share of retrieved content that is relevant to the question | question, context |
| `contextual_precision` | retrieval, rag | whether relevant chunks are ranked above irrelevant ones (rank-weighted) | question, context, reference answer |
| `contextual_recall` | retrieval, rag | share of the reference answer supported by the retrieved context | question, context, reference answer |
| `faithfulness` | rag | whether the answer's claims are supported by the context | question, answer, context |
| `answer_relevancy` | rag | whether the answer addresses the question | question, answer |
| `correctness` | rag (answerable only) | custom `GEval`: is the answer factually consistent with the reference answer | question, answer, reference answer |

## DeepEval vs native metrics

DeepEval has **no** `recall@k`, `hit@k`, `mrr` or `ndcg`. Its "contextual recall" asks whether the context can support the reference answer, and does not check that a specific gold page was retrieved. Its scores come from an LLM, so they depend on the judge and can vary between runs. The hand-written suite is deterministic and free. This suite trades that for answer-level quality judged by an LLM.

## How to run (untested)

The imports resolve with the project root and `src` on `sys.path` (see the top of `scripts/run_retrieval_eval.py`).

```python
from llm.llm import judge_model
from evaluation.deep_eval.rag_eval.suite import run_deepeval_retrieval_suite, run_deepeval_rag_suite

report = run_deepeval_retrieval_suite(judge_model(), strategy="hybrid", limit=5)
```

Start with a small `limit`: every case makes several judge calls per metric, and the free OpenRouter models are often rate-limited or unavailable. `OPENROUTER_API_KEY` must be set in `.env`.

## Known limits

- `measure()` turns a missing score into 0.0 (`metric.score or 0.0`), so an unscored metric looks like a real zero.
- `THRESHOLD` is a single constant for all metrics.
- Judge quality limits what the scores mean: the judge model is a free-tier model, so scores should be checked on a few cases before being trusted.
