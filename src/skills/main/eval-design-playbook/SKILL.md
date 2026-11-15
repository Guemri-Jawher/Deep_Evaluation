---
name: eval-design-playbook
description: >-
  Step-by-step method for answering "how should I evaluate X?" questions: choose the evaluation
  layer (LLM output, RAG, agent, model, production), pick metrics from the four families, and
  structure the answer as dataset, metrics, judge, gate. Use when the user asks how to test, measure,
  benchmark, validate or monitor an LLM app, RAG system, agent or judge.
license: MIT
metadata:
  version: "1.0"
  owner: kb-agent
---

# Eval design playbook

This skill gives the *shape* of a good evaluation answer. Every fact in the answer still has to be
retrieved with `search_docs` and cited (see the `answer-with-citations` skill).

## Step 1: identify the layer

| System under test | Retrieve (queries) |
|-------------------|--------------------|
| Plain LLM output | "deterministic metrics", "LLM as a judge" |
| RAG | "RAG triad", "recall at k", "faithfulness" |
| Agent | "trajectory evaluation", "component metrics", "pass^k" |
| Model choice | "private benchmark", "quality cost latency" |
| Live app | "online evaluation", "guardrails", "drift" |

## Step 2: answer in this order

1. **Dataset** - golden set: happy path, long tail, known failures, adversarial. Versioned in the repo.
2. **Metrics** - one or more from each family: task, quality, safety, operational. See
   [metric families](references/metric-families.md) for the checklist (names only; get definitions from the guide).
3. **Grader** - code first, then a validated LLM judge. State the biases to control
   ([judge checklist](references/judge-checklist.md)).
4. **Loop** - baseline, change one thing, re-measure, decide. Mention the CI gate.

## Step 3: guard against common mistakes

- Scoring only the final answer of a RAG system: decompose retrieval vs generation.
- Trusting an unvalidated judge: require agreement with human labels.
- Reporting a single number: ask for sample size and a confidence interval.

## Output template

```
Recommendation (1-2 sentences)
1. Dataset: ...
2. Metrics: task / quality / safety / operational
3. Grader: code checks + judge (validated how)
4. Loop and gate: ...
Citations inline throughout.
```
