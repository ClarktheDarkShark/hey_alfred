
from typing import Any, Dict

from lang_memgpt.RAG_Structure.chains import retrieval_grader
from lang_memgpt._schemas import State  # Updated to new schema


async def grade_documents(state: State) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state: The current graph state

    Returns:
        state: Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    # Check if 'question' exists in state
    if "question" not in state:
        raise ValueError(
            "Key 'question' is missing from state. Please ensure the input is correctly passed.")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
