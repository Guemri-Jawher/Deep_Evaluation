---
name: claim-verification
description: >-
  Claim-by-claim verification of a draft answer against cited evidence, producing a pass/fail
  verdict with specific issues. Use when asked to check, critique or ground-check a draft answer
  before it is shown to the user.
license: MIT
metadata:
  version: "1.0"
  owner: kb-agent
---

# Claim verification

## Procedure

1. Extract claims: one atomic factual statement per line. Ignore opinions and formatting.
2. Verify each claim:
   - Find the cited chunk in the evidence given in the brief, or with `search_docs`.
   - `supported`: the chunk states the claim (numbers must match exactly).
   - `contradicted`: the chunk says something different.
   - `unsupported`: no chunk states it, or no citation was given.
3. Verify citations: every `[id]` must have been retrieved and must contain the claim it follows.
4. Decide: `pass` only when there are zero issues.

## Output

```
verdict: pass | fail
issues:
- <claim> -> unsupported|contradicted|bad-citation (<why>)
```

Be specific and brief. Do not rewrite the answer and do not add new facts.
