# Judge calibration (2026-11-06T21:04:19+00:00)

- judge: nvidia/nemotron-3.5-bolt-e4b (nvidia)

| rubric | n | parse_rate | exact | within_1 | weighted_kappa | spearman | pass_kappa | mean_bias |
|---|---|---|---|---|---|---|---|---|
| faithfulness | 7 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| correctness | 7 | 1.00 | 0.86 | 1.00 | 0.97 | 0.94 | 1.00 | -0.14 |
| answer_relevance | 6 | 1.00 | 0.83 | 1.00 | 0.97 | 0.95 | 1.00 | -0.17 |

## Bias probes

`{"n_probes": 7, "position": {"position_consistency": 1.0, "first_position_win_rate": 0.5}, "verbosity": {"prefers_concise_correct": 1.0, "n": 5.0}}`
