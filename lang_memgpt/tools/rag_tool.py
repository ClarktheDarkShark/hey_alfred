import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
import csv
from langchain.document_loaders.csv_loader import CSVLoader
import logging
from ..RAG_Structure.nodes.ingestion import ingest_data  # Fix import
from ..RAG_Structure.nodes.retrieve import retrieve

logger = logging.getLogger(__name__)

# Add these constants at the top
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdf_docs")
CSV_DIR = os.path.join(BASE_DIR, "data", "csv_docs")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "vector_store.parquet")

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

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
    embedding_model = OpenAIEmbeddings()

    try:
        if os.path.exists(VECTOR_STORE_PATH):
            vector_store = SKLearnVectorStore(
                embedding=embedding_model,
                persist_path=VECTOR_STORE_PATH,
                serializer="parquet"
            )
        else:
            # Load documents
            documents = []
            for directory, extensions in [(PDF_DIR, ".pdf"), (CSV_DIR, ".csv")]:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if filename.endswith(extensions):
                            file_path = os.path.join(directory, filename)
                            try:
                                loader = PyPDFLoader(file_path) if extensions == ".pdf" else CSVLoader(file_path)
                                documents.extend(loader.load())
                            except Exception as e:
                                logger.error(f"Error loading {filename}: {str(e)}")
                                continue

            # Process documents
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                doc_chunks = text_splitter.split_documents(documents)
                
                vector_store = SKLearnVectorStore.from_documents(
                    documents=doc_chunks,
                    embedding=embedding_model,
                    persist_path=VECTOR_STORE_PATH,
                    serializer="parquet",
                )
                vector_store.persist()
            else:
                return "No documents available to search."

        # Perform search
        retriever = vector_store.as_retriever()
        results = retriever.get_relevant_documents(query)
        
        # Fix the syntax error in this line
        return "\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(results[:3])])
    except Exception as e:
        logger.error(f"Error in document retriever: {str(e)}")
        return f"Error searching documents: {str(e)}"

# @tool
# async def wrapped_ingest_data(*args, **kwargs) -> str:
#     print(f"[RAG TOOL] wrapped_ingest_data called with:", flush=True)
#     print(f"[RAG TOOL] args: {args}", flush=True)
#     print(f"[RAG TOOL] kwargs: {kwargs}", flush=True)
    
#     try:
#         if args and isinstance(args[0], dict):
#             filename = args[0].get('fileName') or args[0].get('file_name')
#             print(f"[RAG TOOL] Extracted filename from dict: {filename}", flush=True)
#         elif args:
#             filename = args[0]
#             print(f"[RAG TOOL] Using direct filename: {filename}", flush=True)
#         else:
#             print("[RAG TOOL] No filename provided in args", flush=True)
#             raise ValueError("No filename provided")

#         print(f"[RAG TOOL] Calling ingest_data with filename: {filename}", flush=True)
#         result = await ingest_data(filename)
#         print(f"[RAG TOOL] ingest_data result: {result}", flush=True)
#         return result
#     except Exception as e:
#         print(f"[RAG TOOL] Error in wrapped_ingest_data: {str(e)}", flush=True)
#         return f"Error processing document: {str(e)}"

# @tool
# async def wrapped_retrieve(*args, **kwargs) -> str:
#     print("[RAG TOOL] wrapped_retrieve called with:", flush=True)
#     print(f"[RAG TOOL] args: {args}", flush=True)
#     print(f"[RAG TOOL] kwargs: {kwargs}", flush=True)
    
#     try:
#         state = None
#         if args and isinstance(args[0], dict):
#             state = args[0].get('state')
#             print(f"[RAG TOOL] Found state in args: {state}", flush=True)
#         elif kwargs and 'state' in kwargs:
#             state = kwargs['state']
#             print(f"[RAG TOOL] Found state in kwargs: {state}", flush=True)
        
#         if not state:
#             print("[RAG TOOL] No state found, creating default", flush=True)
#             state = {"question": "Please provide information about the document"}
        
#         print(f"[RAG TOOL] Calling retrieve with state: {state}", flush=True)
#         result = await retrieve(state)
#         print(f"[RAG TOOL] Retrieve result: {result}", flush=True)
#         return result
#     except Exception as e:
#         print(f"[RAG TOOL] Error in wrapped_retrieve: {str(e)}", flush=True)
#         return f"Error retrieving document: {str(e)}"
