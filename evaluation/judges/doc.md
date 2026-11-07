# How judge evaluation works

A judge is an LLM that scores answers on a 1 to 5 rubric. Before you trust its scores, you have to check them. This folder has two parts: the judge itself (`judge.py`) and the checks on it (`calibration.py`). This doc explains both. For how the RAG suite uses the judges, see `evaluation/hard_coded_lib/doc.md`.

## 1. The two questions

| Question | Check | Data | Code |
|---|---|---|---|
| Does the judge agree with humans? | calibration | `judge_labels.jsonl` | `calibrate` |
| Is the judge biased? | bias probes | `pairwise_probes.jsonl` | `probe_position_bias`, `probe_verbosity_bias` |

Both use data from `datasets/data/v1/judge/` and the loader (`load_cases("judge_labels")` and `load_cases("pairwise_probes")`).

## 2. The data

**`judge_labels.jsonl`:** 35 examples scored 1 to 5 by a human (`annotator` defaults to "author").

| Rubric | Examples | Score 1 / 2 / 3 / 4 / 5 | Pass (4 or 5) | `inputs` fields |
|---|---|---|---|---|
| `faithfulness` | 12 | 3 / 1 / 2 / 1 / 5 | 6 | `context`, `answer` |
| `correctness` | 17 | 4 / 2 / 2 / 2 / 7 | 9 | `question`, `reference`, `answer` |
| `answer_relevance` | 6 | 2 / 1 / 1 / 0 / 2 | 2 | `question`, `answer` |

The `inputs` fields are exactly the `{{variables}}` of the rubric prompt, so a label can be passed straight to `judge.score(**label.inputs)`.

**`pairwise_probes.jsonl`:** 8 pairs, each with a question, a reference, a `good` answer and a `bad` one. 5 have `kind="verbosity"` (a concise correct answer against a long padded one) and 3 have `kind="correctness"`.

## 3. How a judge scores (`judge.py`)

1. `make_judge("judge_<rubric>")` (in `__init__.py`) builds an `LLMJudge` with the configured judge model, the rubric prompt from the registry, and a disk cache.
2. `judge.score(**variables)` fills the prompt, calls the model, and takes the last JSON object with a `score` from the reply. The score is rounded and must be 1 to 5.
3. If parsing fails it asks once more for "ONLY the JSON object". After that the result is `Verdict(None, "unparseable judge output")`.
4. Replies are cached by model name and prompt (`data/cache/judge_cache.jsonl`), so a rerun costs nothing.
5. `Verdict.normalized` maps 1 to 5 onto 0 to 1, and `Verdict.passed` means a score of 4 or more (`PASS_THRESHOLD = 4`).

Pairwise judging (`score_pairwise`, rubric `judge_pairwise`) returns `winner` as "A", "B" or "tie". `compare_pair` runs both orders and returns `first`, `second`, `tie` or `inconsistent`.

## 4. Calibration (`calibrate`)

`calibrate(labels, score_fn)` asks the judge to score each label and compares its score with the human score.

`score_fn(label)` returns the judge's 1 to 5 score, or `None` if it could not be parsed:

```python
rubric = "faithfulness"
judge = make_judge(f"judge_{rubric}")
labels = [l for l in load_cases("judge_labels") if l.rubric == rubric]
report = calibrate(labels, lambda l: judge.score(**l.inputs).score)
```

Always calibrate **one rubric at a time**. A 4 on faithfulness and a 4 on correctness do not mean the same thing, and the pass rate and score spread differ per rubric.

| Field | Meaning | Read it as |
|---|---|---|
| `parse_rate` | share of labels the judge scored | below 1.0 means the output format breaks; unscored labels are left out of the other fields |
| `exact` | same score as the human | strict; with a 5-point scale a good judge is often 1 point off |
| `within_1` | at most 1 point apart | the lenient version of `exact` |
| `weighted_kappa` | agreement on the 1 to 5 scale, chance-corrected, big misses count more (quadratic) | 1 is perfect, 0 is chance level |
| `spearman` | rank correlation | does the judge order answers like the human, even if its scale is shifted |
| `pass_kappa` | Cohen's kappa on pass or fail (score of 4 or more) | the number that matters for gating, since the suites use the pass decision |
| `mean_bias` | judge minus human | positive means lenient, negative means harsh |

`n` counts all labels. With fewer than 2 scored labels the agreement fields are NaN.

A common reading of kappa (Landis and Koch) calls 0.61 to 0.80 substantial agreement and above 0.80 almost perfect. That is a convention, not a rule of this project. Pick the level you require and write it down before you look at the results.

## 5. Bias probes

Both probes call the pairwise judge.

**`probe_position_bias`:** judges each pair twice, with the answers in both orders.
- `position_consistency`: share of pairs where the same answer wins in both orders (a tie in both counts as consistent).
- `first_position_win_rate`: 0.5 means no preference for the first slot.
- It does not measure accuracy: a judge that always answers "tie" is perfectly consistent.

**`probe_verbosity_bias`:** on the `verbosity` probes the concise answer is correct and the long one is padded or wrong.
- `prefers_concise_correct`: share of probes where the concise answer wins in both orders, plus `n`.
- An `inconsistent` result counts as a loss, so it is not separated from a real preference for the long answer.

## 6. Cost

| Step | Calls |
|---|---|
| calibrate all 35 labels | 35 |
| position bias, 8 pairs | 16 |
| verbosity bias, 5 pairs | 10 |

That is about 61 calls. A free tier allows about 50 requests per day, but the disk cache means a rerun of the same inputs is free, so calibrate one rubric per day if you need to.

## 7. Limits to keep in mind

- **Small data.** 35 labels in total and 6 to 17 per rubric. Kappa on 6 examples is very noisy, and the guides suggest 50 to 100 labels (agentic patterns guide) or a few hundred (evaluation guide).
- **One annotator.** Every label has `annotator = "author"`, so there is no measure of how well humans agree with each other. A kappa of 0.7 between judge and author means little if two humans would only agree at 0.7.
- **Bimodal human scores.** Most labels are 1 or 5 (23 of 35), and 4 is rare (3 of 35). Cases near the pass line, the ones that matter, are few.
- **The judge model.** The judge uses the configured judge model, and the cache key includes its name. If you change the model, the cache gives you new calls and new numbers.
- **Reasoning models.** A reasoning judge can spend its token cap on thinking and return no JSON. That shows up as a low `parse_rate`.
- **Probes are few.** 8 pairs cannot separate a small bias from noise.
