# Judge evaluation: plan and checklist

Working notes for the calibration of the LLM judges. **No results yet.** This file says what to measure, what to expect from the data, and what to write down before running anything. How the code works is in `evaluation/judges/doc.md`.

## 1. Goal

Decide whether each judge can be trusted to pass or fail answers in the RAG suite (`evaluation/hard_coded_lib/suites/rag.py`). The suite's pass decision uses only **faithfulness** and **correctness** (with recall and citation checks). `answer_relevance` and `context_relevance` are recorded but never change the verdict, so those two matter less for gating.

## 2. What we have

| Data | Size | Notes |
|---|---|---|
| `judge_labels.jsonl` | 35 | faithfulness 12, correctness 17, answer_relevance 6; no labels for `context_relevance` |
| `pairwise_probes.jsonl` | 8 | 5 verbosity, 3 correctness |
| Human scores | 1:9, 2:4, 3:5, 4:3, 5:14 | mostly extremes; only 3 labels at 4 |
| Annotators | 1 (the author) | no inter-annotator agreement |

## 3. Known weaknesses of the data

1. **Too few labels per rubric.** 6 (answer relevance) to 17 (correctness). Agreement numbers will have wide uncertainty, and the code does not compute an interval for them.
2. **Single annotator.** The judge can at best match one person.
3. **Few labels near the threshold.** The pass line is 4, and only 3 labels score 4. Disagreements between 3 and 4 decide the pass or fail, but the data barely covers them.
4. **No labels for `context_relevance`,** so that judge cannot be calibrated at all.
5. **Probe pairs are few,** and `inconsistent` verdicts are counted as a loss in the verbosity probe.

## 4. Questions to answer, in order

| # | Question | Measure | Rubrics |
|---|---|---|---|
| 1 | Can the judge produce a parseable score? | `parse_rate` | all |
| 2 | Does it agree with the human on pass or fail? | `pass_kappa` | faithfulness, correctness first |
| 3 | Does it agree on the full scale? | `weighted_kappa`, `spearman`, `within_1` | all |
| 4 | Is it lenient or harsh? | `mean_bias` | all |
| 5 | Does it prefer the first slot? | `position_consistency`, `first_position_win_rate` | pairwise |
| 6 | Does it prefer long answers? | `prefers_concise_correct` | pairwise |
| 7 | Does a different judge model change the answers? | same metrics, second model | faithfulness, correctness |

## 5. Decide before looking

Write the acceptance rule down first, so the numbers cannot shape it. A starting suggestion to adjust:

- `parse_rate` at or near 1.0.
- `pass_kappa` at least 0.6 per rubric (a "substantial" level by the usual convention).
- `mean_bias` close to 0 (say within 0.5 points).
- `position_consistency` high, and `first_position_win_rate` near 0.5.

These thresholds are a suggestion, not something the project has fixed. With 6 to 17 labels per rubric, treat a pass as "no evidence of a problem", not as proof.

## 6. Run order

1. Calibrate faithfulness, then correctness, then answer relevance, one rubric per run.
2. Run both probes with the pairwise judge.
3. Repeat 1 and 2 with a second judge model and compare.
4. Inspect the labels where the judge and the human differ by 2 points or more. They may reveal a label error or a rubric that is unclear.

## 7. Results table (to fill)

| Judge model | Rubric | n | parse_rate | exact | within_1 | weighted_kappa | spearman | pass_kappa | mean_bias |
|---|---|---|---|---|---|---|---|---|---|
| | faithfulness | 12 | | | | | | | |
| | correctness | 17 | | | | | | | |
| | answer_relevance | 6 | | | | | | | |

| Judge model | position_consistency | first_position_win_rate | prefers_concise_correct (n=5) |
|---|---|---|---|
| | | | |

## 8. Disagreements to review

| Label id | Rubric | Human | Judge | Why they differ (read the judge reasoning) |
|---|---|---|---|---|
| | | | | |

## 9. Improvements to consider

- More labels, especially scores of 3 and 4 and for `context_relevance`.
- A second annotator on a subset, to measure human agreement.
- A bootstrap interval for the agreement numbers (`evaluation/core/stats.py` already has one for means).
- Separate `inconsistent` from "prefers long" in the verbosity probe.
- A script that runs the calibration and writes a report like the other suites do.


# Notes summary
Question	Metric	Good value
Does it produce a score?	parse_rate	near 1.0
Same score as the human?	exact	strict, often below 0.6
Close to the human?	within_1	high
Agrees on the whole scale?	weighted_kappa	high
Agrees on pass or fail?	pass_kappa	the key one
Lenient or harsh?	mean_bias	near 0
Does the slot matter?	position_consistency, first_position_win_rate	1.0 and 0.5
Does length win?	prefers_concise_correct	1.0
