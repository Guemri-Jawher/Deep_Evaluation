You are grading whether the ANSWER is correct compared with the REFERENCE answer. The reference is ground truth. Wording may differ; meaning must match.

Scale:
5 = same meaning as the reference, all key facts present and correct
4 = correct; omits a minor detail from the reference
3 = partially correct: main point right but a key fact missing or imprecise
2 = mostly incorrect or very incomplete
1 = contradicts the reference or answers a different question

If the reference says the information is unavailable, the answer must also decline; inventing content scores 1.

Instructions:
1. List the key facts in the reference.
2. Check each against the answer.
3. Ignore length, tone and formatting.
4. Output ONLY this JSON on the last line: {"reasoning": "<2-4 sentences>", "score": <1-5>}

QUESTION:
{{question}}

REFERENCE:
{{reference}}

ANSWER:
{{answer}}
