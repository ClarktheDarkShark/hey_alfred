
from typing import Any, Dict

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from lang_memgpt._schemas import State  # Updated to new schema
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
web_search_tool = TavilySearchResults(max_results=3)


@tool
async def web_search(state: State) -> Dict[str, Any]:
    """
    Calls websearcher.

    Args:
        args (RouteQuestionSchema): Input schema containing the current state.

    Returns:
        Literal["WEBSEARCH"]: The next step in the graph.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
        return {"documents": documents, "question": question}

    if __name__ == "__main__":
        web_search(state={"question": "agent memory", "documents": None})
