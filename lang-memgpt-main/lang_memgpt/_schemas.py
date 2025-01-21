from __future__ import annotations

from typing import List, Optional, Dict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel


class GraphConfig(TypedDict):
    model: str | None
    """The model to use for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""


# Define the schema for the state maintained throughout the conversation
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""
    core_memories: List[str]
    """The core memories associated with the user."""
    recall_memories: List[str]
    """The recall memories retrieved for the current context."""
    question: Optional[str] = None
    """The user's query or question."""
    generation: Optional[str] = None
    """The generated response from the LLM."""
    web_search: bool = False
    """Flag to determine if a web search is required."""
    documents: List[str] = []
    """Documents retrieved from RAG for answering the query."""
    graded_documents: Optional[List[str]] = None
    """Optional: Graded documents for relevance scoring."""


class RetrieveInput(BaseModel):
    state: Dict[str, Any]

class IngestInput(BaseModel):
    filename: str

class ToolCallInput(BaseModel):
    name: str
    arguments: Dict[str, Any]


__all__ = [
    "State",
    "GraphConfig",
]