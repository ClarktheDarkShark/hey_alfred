import os
import logging
from dotenv import load_dotenv
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain.tools import tool

load_dotenv()
# Ensure the directory exists
# log_dir = "./logs"

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(message)s", filename=os.path.join(log_dir, "ingestion.log"), filemode="a")

# logging.info("Logging system initialized successfully.")

# Initialize retriever globally to avoid ImportError
retriever = None  # Placeholder to be initialized later


@tool
def ingest_data(command: str) -> str:
    """
    Loads and processes all PDF and CSV files in the 'docs' folder,
    splits them into chunks, saves embeddings to ChromaDB, 
    and generates metadata with file names and counts.

    Args:
        command (str): Pass 'load_docs' to trigger ingestion.

    Returns:
        str: Metadata including counts, file names, and errors.
    """
    try:
        # Configure logging (Moved inside the function)
        # log_dir = "./logs"
        # os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
        # logging.basicConfig(level=logging.INFO,
        #                     format="%(asctime)s - %(levelname)s - %(message)s",
        #                     filename=os.path.join(log_dir, "ingestion.log"),
        #                     filemode="a")
        
        
        # Set up paths
        base_dir = os.path.dirname(__file__)  # Get current script directory
        docs_path = os.path.abspath(os.path.join(
            base_dir, "../../../docs"))  # Move up to root
        print(f"[INGEST] Docs path: {docs_path}", flush=True)

        # Check if docs directory exists
        if not os.path.exists(docs_path):
            return f"Directory does not exist: {docs_path}"

        # Initialize metadata
        pdf_files = []
        csv_files = []
        error_files = []
        docs_list = []

        # Process all files in the 'docs' folder
        print(flush=True)
        for file in os.listdir(docs_path):
            print(f"Processing file: {file}", flush=True)
            file_path = os.path.join(docs_path, file)

            try:
                # Handle PDFs
                if file.endswith(".pdf"):
                    print(f"Detected PDF file: {file_path}", flush=True)
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    docs_list.extend(loaded_docs)
                    print(f"Successfully loaded PDF file: {file_path} with {len(loaded_docs)} documents", flush=True)
                    pdf_files.append(file)

                # Handle CSVs
                elif file.endswith(".csv"):
                    print(f"Detected CSV file: {file_path}", flush=True)
                    loader = CSVLoader(file_path)
                    loaded_docs = loader.load()
                    docs_list.extend(loaded_docs)
                    print(f"Successfully loaded CSV file: {file_path} with {len(loaded_docs)} documents", flush=True)
                    csv_files.append(file)

                else:
                    print(f"Skipping unsupported file type: {file}", flush=True)

            except Exception as e:
                print(f"Failed to process file {file}: {str(e)}", flush=True)
                # Uncomment the line below to collect error details if needed
                # error_files.append(f"{file}: {str(e)}")

        print(flush=True)
        # Exit early if no valid files were found
        if not docs_list:
            return "No valid documents found in the 'docs' folder."

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Create or update our local Chroma vector store
        try:
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding=OpenAIEmbeddings(),
                persist_directory="/Users/.chroma"  # Replaced "./.chroma"
            )
            vectorstore.persist()
            print(
                "Vectorstore created and persisted at: /Users/.chroma")

            # Initialize retriever globally
            global retriever
            retriever = vectorstore.as_retriever()

        except TypeError as te:
            print(f"TypeError during vector store creation: {str(te)}")
            return f"Failed to process embeddings due to a TypeError: {str(te)}"

        except Exception as e:
            print(
                f"Unexpected error during vector store creation: {str(e)}")
            return f"An unexpected error occurred while creating the vector store: {str(e)}"

        # Generate metadata summary
        metadata = {
            "pdf_files_processed": len(pdf_files),
            "csv_files_processed": len(csv_files),
            "errors": len(error_files),
            "total_files_processed": len(pdf_files) + len(csv_files),
            "processed_files": {
                "pdf_files": pdf_files,
                "csv_files": csv_files,
            },
            "errors_details": error_files,
        }

        # Return metadata summary
        print("Ingestion complete with metadata: %s", metadata)
        return f"Ingestion complete! Metadata: {metadata}"

    except Exception as e:
        # Catch any unexpected errors during the entire function execution
        print(f"Critical error during ingestion: {str(e)}")
        return f"An error occurred during ingestion: {str(e)}"


if __name__ == "__main__":
    # For direct execution, simulate the agent call
    result = ingest_data.invoke("load_docs")
    print(result)  # Output metadata to terminal
    print("Ingestion script executed successfully.")





# import os
# import logging
# import json
# from dotenv import load_dotenv
# from typing import List
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, CSVLoader
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain.tools import tool, StructuredTool

# load_dotenv()

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )

# # -------------------------------------------------------------------
# # IMPORTANT: Ensure both ingestion & retrieval point to the SAME folder
# # -------------------------------------------------------------------
# # Update the persist directory to match the API
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DATA_DIR = os.path.join(BASE_DIR, "data")
# PERSIST_DIRECTORY = os.path.join(DATA_DIR, ".chroma")
# PDF_DIR = os.path.join(DATA_DIR, "pdf_docs")
# CSV_DIR = os.path.join(DATA_DIR, "csv_docs")
# COLLECTION_NAME = "rag-chroma"  # Add this constant

# # Ensure directories exist
# os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
# os.makedirs(PDF_DIR, exist_ok=True)
# os.makedirs(CSV_DIR, exist_ok=True)

# @tool  # Explicitly mark as async
# async def ingest_data(filename: str) -> str:
#     """
#     Ingest a single PDF or CSV into the Chroma vector store.

#     Args:
#         filename (str): Name of the file in either pdf_docs or csv_docs.
#     Returns:
#         str: A success or error message.
#     """
#     filename = filename.strip()  # Default to the string itself
#     print(f"[INGEST] filename: {filename}", flush=True)
#     try:
#         if isinstance(filename, str):
#             try:
#                 maybe_json = json.loads(filename)
#                 if isinstance(maybe_json, dict):
#                     filename = maybe_json.get("fileName") or maybe_json.get("file_name")
#             except json.JSONDecodeError:
#                 pass  # Input is not JSON, use as raw string
#     except Exception as e:
#         print(f"[INGEST] Error parsing JSON: {str(e)}", flush=True)
#         return f"Error parsing JSON: {str(e)}"
        
#     print(f"[INGEST] Starting ingestion with PERSIST_DIRECTORY: {PERSIST_DIRECTORY}", flush=True)
#     print(f"[INGEST] File to process: {filename}", flush=True)
#     print(f"[INGEST] PDF_DIR: {PDF_DIR}", flush=True)
#     print(f"[INGEST] CSV_DIR: {CSV_DIR}", flush=True)
    
#     logger.info(f"ingest_data called with filename: {filename}")
#     try:
#         pdf_path = os.path.join(PDF_DIR, filename)
#         csv_path = os.path.join(CSV_DIR, filename)
#         print(f"[INGEST] Full paths being checked:", flush=True)
#         print(f"[INGEST] PDF path: {pdf_path} exists: {os.path.exists(pdf_path)}", flush=True)
#         print(f"[INGEST] CSV path: {csv_path} exists: {os.path.exists(csv_path)}", flush=True)

#         if os.path.exists(pdf_path):
#             print(f"[INGEST] Found PDF file at: {pdf_path}", flush=True)
#             file_path = pdf_path
#             loader = PyPDFLoader(file_path)
#         elif os.path.exists(csv_path):
#             print(f"[INGEST] Found CSV file at: {csv_path}", flush=True)
#             file_path = csv_path
#             loader = CSVLoader(file_path)
#         else:
#             print(f"[INGEST] File not found in either directory", flush=True)
#             return f"File not found in pdf_docs or csv_docs: {filename}"

#         print(f"[INGEST] Using file path: {file_path}", flush=True)
#         print(f"[INGEST] File exists check: {os.path.exists(file_path)}", flush=True)
#         print(f"[INGEST] File size: {os.path.getsize(file_path)} bytes", flush=True)

#         print(f"[INGEST] Loading document content", flush=True)
#         documents = loader.load()
#         print(f"[INGEST] Loaded {len(documents)} document(s)", flush=True)
        
#         print(f"[INGEST] Splitting text into chunks", flush=True)
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         doc_chunks = text_splitter.split_documents(documents)
#         print(f"[INGEST] Created {len(doc_chunks)} chunks", flush=True)

#         print(f"[INGEST] Creating vector store at {PERSIST_DIRECTORY}", flush=True)
#         vectorstore = Chroma.from_documents(
#             documents=doc_chunks,
#             embedding=OpenAIEmbeddings(),
#             collection_name=COLLECTION_NAME,
#             persist_directory=PERSIST_DIRECTORY,
#         )
#         print("[INGEST] Persisting vector store", flush=True)
#         vectorstore.persist()
#         print("[INGEST] Vector store created and persisted successfully", flush=True)

#         return f"Successfully ingested {filename} into {COLLECTION_NAME}"
#     except Exception as e:
#         print(f"[INGEST] Error during ingestion: {str(e)}", flush=True)
#         logger.error(f"Error ingesting {filename}: {str(e)}")
#         return f"Error processing file: {str(e)}"

# @tool  # Explicitly mark as async
# async def wrapped_ingest_data(tool_input: str) -> str:
#     """
#     Ingest a document into the Chroma vector store. Expects a filename or JSON string containing the filename.
    
#     Parameters:
#         tool_input (str): A filename or JSON string with keys "fileName" or "file_name".
    
#     Returns:
#         str: Success or error message.
#     """

#     print('[WRAPPED INGEST] wrapped_ingest_data called with:', flush=True)
#     filename = tool_input.strip()  # Default to the string itself
#     print(f"[WRAPPED INGEST] filename: {filename}", flush=True)
#     try:
#         if isinstance(tool_input, str):
#             try:
#                 maybe_json = json.loads(tool_input)
#                 if isinstance(maybe_json, dict):
#                     filename = maybe_json.get("fileName") or maybe_json.get("file_name")
#             except json.JSONDecodeError:
#                 pass  # Input is not JSON, use as raw string

#         if not filename:
#             raise ValueError("Invalid tool_input: Missing or empty filename.")

#         logger.info(f"Processing file: {filename}")
#         return await ingest_data.ainvoke(filename)
#     except Exception as e:
#         logger.error(f"Error in wrapped_ingest_data: {str(e)}")
#         return f"Error processing document: {str(e)}"
