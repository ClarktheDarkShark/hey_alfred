# lang-memgpt-main/lang_memgpt/RAG_Structure/nodes/generate.py
from typing import Any, Dict

from lang_memgpt.RAG_Structure.chains.generation import generation_chain
from lang_memgpt._schemas import State  # Updated to new schema


async def generate(state: State) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke(
        {"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
