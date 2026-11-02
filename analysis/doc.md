1. Retrieval metrics (the ones in your reports)
Each question has gold evidence, and the metrics are scored on the chunks returned by the retriever.

Metric	Question it answers	How it is computed	Range	Key point
hit@k	Did I find anything relevant?	1 if any top-k chunk is relevant, else 0	0 or 1	Most lenient. One relevant chunk is enough.
recall@k (k = 1, 3, 5, 10)	Did I find everything needed?	evidence units found in the top k / total units	0 to 1	The main metric. Counts units, so a 2-source question scores 0.5 with one source.
precision@5	Is what I retrieved clean?	relevant chunks in the top 5 / 5	0 to 1	Low (about 0.2) by design, since most questions have only 1-2 relevant chunks. Not useful for judging.
mrr (reciprocal rank)	How high is the first relevant chunk?	1 / rank of the first relevant chunk	0 to 1	Order-sensitive, but looks only at the first hit and has no cutoff.
ndcg@5	How well ordered is the whole top 5?	position-discounted gain, divided by the ideal ordering	0 to 1	Order-sensitive over all relevant chunks. It is the only metric that uses the corpus (via total_relevant).
pass_rate	On how many questions did I succeed fully?	share of questions with recall@5 = 1.0	0 to 1	A pass/fail version of recall@5. It differs only on multi-evidence questions.
Key differences

hit vs recall: hit is satisfied by one relevant chunk, recall needs all the evidence units.
recall vs precision: recall penalizes missing evidence, precision penalizes noise.
mrr vs ndcg: both reward ranking relevant chunks higher. mrr looks at the first relevant chunk only, and ndcg looks at all relevant chunks in the top k.
recall@5 vs recall@10: a large gap means the evidence is found but ranked too low.
2. Operational metrics (computed by summarize)
Metric	Meaning
error_rate	share of cases that crashed
latency_p50_s, latency_p95_s	median and slow-tail latency per case
tokens_per_case	average token use
The retrieval suite turns these off (include_ops=False), so you only see error_rate in its reports.

3. Statistics for uncertainty and comparison
Function	Purpose
bootstrap_ci	gives the [ci_low, ci_high] range shown next to each mean
paired_bootstrap_delta	the difference between two runs on the same cases, with a range. If the range excludes 0, the change is more than noise.
wilson_interval	range for a pass rate, reliable with few cases
mean, percentile	basic helpers (for example p50 and p95)
4. Agent reliability (defined, not used by the retrieval run)
Metric	Meaning
pass@k	probability that at least one of k attempts succeeds
pass^k	probability that all k attempts succeed. It falls fast as k grows.
Here k is the number of attempts, not the top-k chunks.

5. Judge validation (defined, not used by the retrieval run)
These check whether an LLM judge agrees with human labels.

Metric	Meaning
accuracy	share of identical labels
cohens_kappa	agreement on categories (such as pass/fail), corrected for chance
weighted_kappa	same for ordinal scores such as 1-5. A 1-vs-5 disagreement counts more than 1-vs-2.
spearman	rank correlation between two score lists
classification_report	precision, recall, F1, accuracy and tp/fp/fn/tn for a binary label
The precision and recall in classification_report are classification metrics. They are not the retrieval precision@k and recall@k above.

One-line summary
hit asks "found anything?", recall asks "found everything?", precision asks "is it clean?", mrr asks "how high is the first?", and ndcg asks "how good is the whole ordering?". The ranges around them tell you whether a difference between two runs is real.