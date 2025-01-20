import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
import csv
from langchain.document_loaders.csv_loader import CSVLoader

# Async Tool for RAG with Parquet Storage


@tool
async def document_retriever(query: str) -> str:
    """
    Query PDF and CSV documents stored in respective folders using SKLearn Vector DB with Parquet serialization.

    Args:
        query (str): The search query for retrieving relevant document content.
    Returns:
        str: Top matches from the PDF and CSV documents.
    """
    # Paths and Embedding Setup
    data_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "../data/pdf_docs"))  # replaced "./data/pdf_docs"
    csv_data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../data/csv_docs"))
    persist_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "../data/vector_store/vector_store.parquet"))

    # Ensure vector storage directory exists
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)

    embedding_model = OpenAIEmbeddings()

    # Check if vector store already exists
    if os.path.exists(persist_path):
        # Load existing vector store
        vector_store = SKLearnVectorStore(
            embedding=embedding_model,
            persist_path=persist_path,
            serializer="parquet"
        )
    else:
        # Load and Process PDFs
        documents = []
        for file_name in os.listdir(data_path):
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(data_path, file_name))
                documents.extend(loader.load())

        # Load and Process CSVs
        for file_name in os.listdir(csv_data_path):
            if file_name.endswith(".csv"):
                loader = CSVLoader(os.path.join(csv_data_path, file_name))
                documents.extend(loader.load())

        # Split Documents into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        doc_chunks = text_splitter.split_documents(documents)

        # Create and Save Vector Store with Parquet Serializer
        vector_store = SKLearnVectorStore.from_documents(
            documents=doc_chunks,
            embedding=embedding_model,
            persist_path=persist_path,
            serializer="parquet",  # Uses Parquet instead of Pickle
        )
        vector_store.persist()

    # Perform the Query
    retriever = vector_store.as_retriever()
    results = retriever.get_relevant_documents(query)

    # Format Results
    response = "\n".join(
        [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(results[:3])])
    return response if response else "No relevant information found."
