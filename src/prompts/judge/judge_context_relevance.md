You are grading whether the retrieved CONTEXT is relevant to the QUESTION: does it contain the information needed to answer, with little noise?

Scale:
5 = the context contains everything needed; almost all of it is on topic
4 = contains what is needed; some irrelevant passages
3 = contains part of what is needed, or is mostly noise around the answer
2 = only tangentially related
1 = unrelated to the question

Instructions:
1. State what information the question needs.
2. For each context passage say whether it helps, quoting the relevant words.
3. Output ONLY this JSON on the last line: {"reasoning": "<2-4 sentences>", "score": <1-5>}

QUESTION:
{{question}}

CONTEXT:
{{context}}
