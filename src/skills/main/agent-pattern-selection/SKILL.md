---
name: agent-pattern-selection
description: >-
  Decision procedure for recommending an agent architecture (single call, prompt chain, routing,
  ReAct, planning, reflection, supervisor-worker, human-in-the-loop) from the user's requirements,
  starting from the simplest option and justifying each step up. Use when the user asks which
  pattern, workflow or agent design to use, or whether they need an agent or multi-agent system.
license: MIT
metadata:
  version: "1.0"
  owner: kb-agent
---

# Agent pattern selection

Retrieve every claim about a pattern with `search_docs` and cite it. This skill only fixes the
order of reasoning.

## Procedure

1. Ask (or infer) three things: can success be defined, can the agent verify its own progress,
   can the worst-case run be afforded? If any answer is no, recommend a workflow first
   (retrieve: "three questions before you build an agent").
2. Walk the ladder from the left and stop at the first rung that fits:
   single call -> prompt chain -> routing -> ReAct agent -> planning -> reflection ->
   supervisor-worker. Retrieve "when to build an agent" for the table.
3. For the chosen pattern retrieve: how it works, where it breaks, cost.
4. Add the controls that are not optional in production: termination budgets, guardrails,
   human-in-the-loop for irreversible actions (retrieve "control patterns").
5. State what you would measure to prove the choice (retrieve "agent eval stack").

## Output template

```
Recommended: <pattern> (rung N of the ladder)
Why not simpler: ...
Why not more complex: ...
Failure modes and fixes: ...
Controls: ...
How to evaluate: ...
```

## Cheat sheet use

Retrieve "pattern selection cheat sheet" to compare cost levels; quote the cost column rather than
guessing. Multi-agent systems are the most expensive rung; require evidence before recommending them.
