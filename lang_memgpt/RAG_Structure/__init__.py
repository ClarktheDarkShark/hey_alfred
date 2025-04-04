# lang-memgpt-main/lang_memgpt/RAG_Structure/__init__.py

from .nodes.ingestion import ingest_data  # Add ingestion
# Import the main RAG tool
from .nodes.retrieve import retrieve  # Entry point as a tool

# Import internal components for graph building
from .nodes.grade_documents import grade_documents
# Removed the import of 'generate' to prevent circular dependency.
from .nodes.web_search import web_search

# Expose public API
__all__ = [
    "ingest_data",
    "retrieve",            # Main tool exposed for agent calls
    "grade_documents",     # Internal nodes for conditional graph flow
    # "generate",         # Omit 'generate' here to avoid circular import
    "web_search",
]
