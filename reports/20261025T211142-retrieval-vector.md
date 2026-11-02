# retrieval-vector (2026-10-25T21:11:42+00:00)

- dataset: v1, cases: 49
- config: `{"strategy": "vector", "split": null, "k": 10, "chunks": 194}`

| metric | mean | 95% CI | n |
|---|---|---|---|
| error_rate | 0.000 | [0.000, 0.000] | 49 |
| hit@1 | 0.653 | [0.510, 0.776] | 49 |
| hit@10 | 0.959 | [0.898, 1.000] | 49 |
| hit@3 | 0.816 | [0.694, 0.918] | 49 |
| hit@5 | 0.898 | [0.816, 0.980] | 49 |
| mrr | 0.754 | [0.651, 0.851] | 49 |
| ndcg@5 | 0.758 | [0.662, 0.847] | 49 |
| pass_rate | 0.898 | [0.816, 0.980] | 49 |
| precision@5 | 0.212 | [0.184, 0.241] | 49 |
| recall@1 | 0.633 | [0.500, 0.765] | 49 |
| recall@10 | 0.949 | [0.887, 1.000] | 49 |
| recall@3 | 0.816 | [0.694, 0.918] | 49 |
| recall@5 | 0.898 | [0.816, 0.980] | 49 |

## Failures (5)

- `r001` [factual] fail=evidence not in top 5
- `r031` [procedure] fail=evidence not in top 5
- `r034` [procedure] fail=evidence not in top 5
- `r035` [numeric] fail=evidence not in top 5
- `r049` [multi_hop] fail=evidence not in top 5
