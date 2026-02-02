# rag_pipeline/run.py

from rag_pipeline.pipeline import build_retriever, generate_answer, needs_retrieval


_retriever = None


def run(query: str) -> str:
    global _retriever

    if _retriever is None:
        _retriever = build_retriever(query)

    if needs_retrieval(query):
        docs = _retriever.invoke(query)
        return generate_answer(query, docs)

    return generate_answer(query, [])
