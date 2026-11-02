You are grading an AI agent's TRAJECTORY (the sequence of tool calls) for one task. You are not grading the final answer.

Rubric (score each 1-5, then give the overall as the lowest of the three):
1. Tool choice: did it pick appropriate tools and avoid unnecessary ones?
2. Arguments: were queries and arguments specific and well-formed (no copy-pasting the whole question when focused queries were possible)?
3. Efficiency and recovery: no repeated identical calls, no wandering; when a tool failed or returned weak results, did it adapt (rewrite query, change tool)?

Scale for each: 5 = ideal, 3 = acceptable with waste, 1 = broken (wrong tools, loops, ignored errors).

Instructions:
1. Walk the steps in order and find the FIRST questionable step, if any.
2. Attribute it: bad tool choice, bad arguments, or bad reasoning.
3. Output ONLY this JSON on the last line: {"reasoning": "<first questionable step and why, 2-4 sentences>", "score": <1-5 overall>}

TASK:
{{task}}

EXPECTED BEHAVIOUR:
{{expectation}}

TRAJECTORY:
{{trajectory}}
