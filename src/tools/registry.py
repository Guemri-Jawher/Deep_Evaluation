from __future__ import annotations

from langchain_core.tools import BaseTool

from rag.retriever import Retriever
from tools.docs import make_search_docs
from tools.math_tools import calculator, pass_k
from tools.notes import save_study_notes


def build_tools(retriever: Retriever) -> list[BaseTool]:
    return [make_search_docs(retriever), calculator(), pass_k(), save_study_notes()]
