You are grading whether the ANSWER is fully supported by the CONTEXT. Ignore whether the answer is correct in the real world; only the context counts.

Scale:
5 = every factual claim is directly supported by the context
4 = all material claims supported; one trivial detail unsupported
3 = mostly supported, one minor unsupported or unverifiable detail
2 = several unsupported claims, or one important unsupported claim
1 = contains claims that contradict the context or are absent from it

An answer that correctly says the context does not contain the information scores 5.

Instructions:
1. List each factual claim in the ANSWER.
2. Mark each as supported or unsupported, quoting the context that supports it.
3. Do not reward length, formatting or confident tone.
4. Then output ONLY this JSON on the last line: {"reasoning": "<2-4 sentences>", "score": <1-5>}

CONTEXT:
{{context}}

ANSWER:
{{answer}}
