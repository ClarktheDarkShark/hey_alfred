from typing import Any, Dict
from lang_memgpt._schemas import State
from lang_memgpt.RAG_Structure.nodes.ingestion import retriever
from langchain.tools import tool  # or your custom tool decorator

# Import Chroma and OpenAI Embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Reload the retriever directly from the saved Chroma vector store
# Replaced --> "./chroma"  # Path where the vectorstore is saved
persist_directory = "/Users/.chroma"
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(),
    # Ensure this matches the collection name used in ingestion
    collection_name="rag-chroma"
)

# Initialize retriever from vectorstore
retriever = vectorstore.as_retriever()


@tool
def retrieve(state: State) -> Dict[str, Any]:
    """
    Retrieves documents from the previously ingested vector store
    based on the user's current question in the 'state' object.
    """
    print("---RETRIEVE---")
    question = state["question"]

    # If you prefer the .invoke() style (like your screenshot):
    documents = retriever.invoke(question)

    return {
        "documents": documents,
        "question": question
    }
