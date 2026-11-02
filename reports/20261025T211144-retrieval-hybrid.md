# retrieval-hybrid (2026-10-25T21:11:44+00:00)

- dataset: v1, cases: 49
- config: `{"strategy": "hybrid", "split": null, "k": 10, "chunks": 194}`

| metric | mean | 95% CI | n |
|---|---|---|---|
| error_rate | 0.000 | [0.000, 0.000] | 49 |
| hit@1 | 0.714 | [0.571, 0.837] | 49 |
| hit@10 | 0.939 | [0.857, 1.000] | 49 |
| hit@3 | 0.878 | [0.776, 0.959] | 49 |
| hit@5 | 0.898 | [0.796, 0.980] | 49 |
| mrr | 0.798 | [0.697, 0.886] | 49 |
| ndcg@5 | 0.778 | [0.681, 0.863] | 49 |
| pass_rate | 0.878 | [0.776, 0.959] | 49 |
| precision@5 | 0.212 | [0.184, 0.245] | 49 |
| recall@1 | 0.684 | [0.551, 0.806] | 49 |
| recall@10 | 0.939 | [0.857, 1.000] | 49 |
| recall@3 | 0.867 | [0.765, 0.959] | 49 |
| recall@5 | 0.888 | [0.796, 0.969] | 49 |

## Failures (6)

- `r001` [factual] fail=evidence not in top 5
- `r029` [factual] fail=evidence not in top 5
- `r032` [explanation] fail=evidence not in top 5
- `r034` [procedure] fail=evidence not in top 5
- `r035` [numeric] fail=evidence not in top 5
- `r049` [multi_hop] fail=evidence not in top 5
