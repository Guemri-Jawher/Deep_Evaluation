"""Run the retrieval layer live (no LLM / API key needed) and print the top chunks per strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.retriever import STRATEGIES, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="what is pass^k?")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--strategy", choices=STRATEGIES, help="default: compare all strategies")
    parser.add_argument("--source", help="restrict to a doc alias, e.g. eval-guide")
    parser.add_argument("--rebuild", action="store_true", help="force re-embedding the corpus")
    args = parser.parse_args()

    retriever = build_index(rebuild=args.rebuild)
    print(f"{len(retriever.chunks)} chunks from sources: {', '.join(retriever.sources)}\n")

    for strategy in [args.strategy] if args.strategy else STRATEGIES:
        print(f"=== {strategy} ===")
        for r in retriever.retrieve(args.query, k=args.k, source=args.source, strategy=strategy):
            snippet = r.chunk.text[:160].replace("\n", " ")
            print(f"{r.rank}. {r.score:.4f}  [{r.chunk.id}]  {snippet}")
        print()


if __name__ == "__main__":
    main()
