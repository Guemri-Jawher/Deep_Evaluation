"""Run the full RAG pipeline (retrieve -> LLM -> cited answer) and send the trace to Langfuse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.llm import agent_model
from observability.langfuse_client import current_trace_id, flush, get_langfuse, trace_run
from rag.pipeline import RagPipeline
from rag.retriever import STRATEGIES, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What is pass^k and why does it matter?")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--strategy", choices=STRATEGIES)
    args = parser.parse_args()

    print(f"Langfuse tracing: {'ON' if get_langfuse() else 'OFF (keys missing in .env)'}")
    pipeline = RagPipeline(build_index(), agent_model(), k=args.k, strategy=args.strategy)

    with trace_run("rag-demo", input=args.question, tags=["demo", "rag"]) as span:
        result = pipeline.answer(args.question)
        trace_id = current_trace_id()
        if span is not None:
            span.update(output=result.answer)

    print(f"\nQ: {result.question}\n\nA: {result.answer}\n")
    print(f"prompt: {result.prompt.tag} ({result.prompt.source})")
    print(f"cited:  {result.cited_ids}")
    print("retrieved:")
    for r in result.retrieved:
        print(f"  {r.rank}. {r.score:.4f} [{r.chunk.id}]")
    if trace_id:
        print(f"\ntrace id: {trace_id}")

    flush()


if __name__ == "__main__":
    try:
        main()
    finally:
        flush()
