# retrieval-hybrid (2026-10-25T21:24:07+00:00)

- dataset: v1, cases: 41
- config: `{"strategy": "hybrid", "split": "dev", "k": 10, "chunks": 194}`

| metric | mean | 95% CI | n |
|---|---|---|---|
| error_rate | 0.000 | [0.000, 0.000] | 41 |
| hit@1 | 0.683 | [0.537, 0.805] | 41 |
| hit@10 | 0.927 | [0.829, 1.000] | 41 |
| hit@3 | 0.854 | [0.732, 0.951] | 41 |
| hit@5 | 0.878 | [0.780, 0.976] | 41 |
| mrr | 0.774 | [0.665, 0.874] | 41 |
| ndcg@5 | 0.751 | [0.643, 0.850] | 41 |
| pass_rate | 0.854 | [0.732, 0.951] | 41 |
| precision@5 | 0.215 | [0.180, 0.249] | 41 |
| recall@1 | 0.646 | [0.500, 0.780] | 41 |
| recall@10 | 0.927 | [0.829, 1.000] | 41 |
| recall@3 | 0.841 | [0.720, 0.939] | 41 |
| recall@5 | 0.866 | [0.756, 0.951] | 41 |

## Failures (6)

- `r001` [factual] fail=evidence not in top 5
- `r029` [factual] fail=evidence not in top 5
- `r032` [explanation] fail=evidence not in top 5
- `r034` [procedure] fail=evidence not in top 5
- `r035` [numeric] fail=evidence not in top 5
- `r049` [multi_hop] fail=evidence not in top 5
