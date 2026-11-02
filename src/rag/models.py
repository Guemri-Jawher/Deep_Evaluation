from __future__ import annotations

import re
from dataclasses import dataclass, field

CHUNK_ID_RE = re.compile(r"^[a-z0-9-]+:p\d+:c\d+$")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc-alias>:p<page>:c<index>", stable for a given chunker config
    text: str
    source: str  # doc alias, e.g. "eval-guide"
    page: int  # 1-based PDF page
    index: int  # chunk index within the page

    def citation(self) -> str:
        return f"[{self.id}]"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int  # 1-based
    method: str = "vector"
    extras: dict[str, float] = field(default_factory=dict)


def extract_citations(text: str) -> list[str]:
    """Return every chunk id cited as `[id]` or `[id1, id2]` in order of appearance."""
    out: list[str] = []
    for group in re.findall(r"\[([^\[\]]+)\]", text):
        for token in re.split(r"[,\s;]+", group.strip()):
            if CHUNK_ID_RE.match(token):
                out.append(token)
    return out
