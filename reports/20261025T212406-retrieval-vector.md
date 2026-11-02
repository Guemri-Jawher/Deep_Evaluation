# retrieval-vector (2026-10-25T21:24:06+00:00)

- dataset: v1, cases: 41
- config: `{"strategy": "vector", "split": "dev", "k": 10, "chunks": 194}`

| metric | mean | 95% CI | n |
|---|---|---|---|
| error_rate | 0.000 | [0.000, 0.000] | 41 |
| hit@1 | 0.634 | [0.488, 0.780] | 41 |
| hit@10 | 0.951 | [0.878, 1.000] | 41 |
| hit@3 | 0.780 | [0.634, 0.902] | 41 |
| hit@5 | 0.878 | [0.780, 0.951] | 41 |
| mrr | 0.734 | [0.615, 0.842] | 41 |
| ndcg@5 | 0.737 | [0.619, 0.836] | 41 |
| pass_rate | 0.878 | [0.780, 0.951] | 41 |
| precision@5 | 0.215 | [0.180, 0.249] | 41 |
| recall@1 | 0.610 | [0.463, 0.756] | 41 |
| recall@10 | 0.939 | [0.866, 1.000] | 41 |
| recall@3 | 0.780 | [0.634, 0.902] | 41 |
| recall@5 | 0.878 | [0.780, 0.951] | 41 |

## Failures (5)

- `r001` [factual] fail=evidence not in top 5
- `r031` [procedure] fail=evidence not in top 5
- `r034` [procedure] fail=evidence not in top 5
- `r035` [numeric] fail=evidence not in top 5
- `r049` [multi_hop] fail=evidence not in top 5
