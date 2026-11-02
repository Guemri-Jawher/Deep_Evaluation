You compare two answers (A and B) to the same QUESTION and pick the better one.

Criteria, in order: factual correctness against the REFERENCE, completeness, then concision. Do NOT prefer an answer because it is longer, appears first, or is more confident. If both are equally good, answer "tie".

Instructions:
1. Check each answer against the reference facts.
2. Note the decisive difference.
3. Output ONLY this JSON on the last line: {"reasoning": "<2-3 sentences>", "winner": "A" | "B" | "tie"}

QUESTION:
{{question}}

REFERENCE:
{{reference}}

ANSWER A:
{{answer_a}}

ANSWER B:
{{answer_b}}
