# api.py
import os
import sys
import logging
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from lang_memgpt.graph import process_chat  # Ensure this path is correct in your project
from lang_memgpt import _schemas as schemas
from lang_memgpt.RAG_Structure.nodes.ingestion import ingest_data

# Set up paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LANG_MEMGPT_DIR = os.path.join(ROOT_DIR, "lang_memgpt")
DATA_DIR = os.path.join(LANG_MEMGPT_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdf_docs")
CSV_DIR = os.path.join(DATA_DIR, "csv_docs")
PERSIST_DIRECTORY = os.path.join(DATA_DIR, ".chroma")

print("[API] Directory structure:", flush=True)
print(f"[API] ROOT_DIR: {ROOT_DIR}", flush=True)
print(f"[API] LANG_MEMGPT_DIR: {LANG_MEMGPT_DIR}", flush=True)
print(f"[API] DATA_DIR: {DATA_DIR}", flush=True)
print(f"[API] PDF_DIR: {PDF_DIR}", flush=True)
print(f"[API] CSV_DIR: {CSV_DIR}", flush=True)
print(f"[API] PERSIST_DIRECTORY: {PERSIST_DIRECTORY}", flush=True)

# Create directories if they do not exist
for dir_path in [LANG_MEMGPT_DIR, DATA_DIR, PDF_DIR, CSV_DIR, PERSIST_DIRECTORY]:
    os.makedirs(dir_path, exist_ok=True)
    print(f"[API] Created/verified directory: {dir_path}", flush=True)

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Run any startup tasks if needed."""
    pass

# Configure CORS (allow all origins for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    configurable: Dict[str, Any]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Process any file upload messages from the conversation.
    for idx, msg in enumerate(request.messages):
        content = msg.get("content", "")
        if msg.get("role") == "user" and content.startswith("File uploaded:"):
            try:
                # Expected format:
                # File uploaded: filename.ext
                # Content: data:<mime>;base64,<base64_string>
                parts = content.split('\nContent:')
                if len(parts) != 2:
                    logger.error("Invalid file upload message format.")
                    continue

                header = parts[0].strip()
                file_data = parts[1].strip()

                # Extract filename (everything after "File uploaded:")
                file_name = header.replace("File uploaded:", "").strip()

                # Remove Data URL prefix if present (e.g., "data:application/pdf;base64,")
                if file_data.startswith("data:"):
                    comma_index = file_data.find(',')
                    if comma_index != -1:
                        file_data = file_data[comma_index+1:]

                file_bytes = base64.b64decode(file_data)

                # Determine the docs path (adjusting path as needed for your project structure)
                base_dir = os.path.dirname(__file__)  # Current script directory
                app_dir = os.path.abspath(os.path.join(base_dir, "../app"))  # Navigate to the app directory
                docs_path = os.path.join(app_dir, "docs")  # Set the docs path to app/docs
                if not os.path.exists(docs_path):
                    os.makedirs(docs_path, exist_ok=True)

                file_save_path = os.path.join(docs_path, file_name)
                with open(file_save_path, "wb") as f:
                    f.write(file_bytes)

                print(f"Saved file {file_name} to {file_save_path} for ingestion.", flush=True)

                request.messages.append({
                    "role": "system",
                    "content": f"load_docs: {file_name}"
                })
                # # Now, update the message so that the agent sees only the file name.
                # # For example, change the message to "File uploaded: filename.ext"
                request.messages[idx]["content"] = f"File uploaded: {file_name}"

            except Exception as e:
                logger.error(f"Error processing file upload message: {str(e)}")
                raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
    
    try:
        logger.info(f"Received chat request: messages={request.messages} configurable={request.configurable}")
        print(f"Received chat request: messages={request.messages} configurable={request.configurable}", flush=True)
        
        # Build configuration for the graph/LLM with already_ingested flag
        config: schemas.GraphConfig = {
            "configurable": {
                "user_id": request.configurable.get("user_id", "default-user"),
                "model": request.configurable.get("model", "gpt-4o")
            }
        }
        
        # Set already_ingested to True if this is a follow-up question about a document
        last_message = request.messages[-1] if request.messages else {"content": ""}
        previous_messages = request.messages[:-1] if len(request.messages) > 1 else []
        
        # Check if this is a follow-up about a previously uploaded document
        is_doc_query = any(word in last_message.get("content", "").lower() for word in ["document", "file", "pdf"])
        has_previous_upload = any("File uploaded:" in msg.get("content", "") for msg in previous_messages)
        
        config["already_ingested"] = is_doc_query and has_previous_upload
        logger.info(f"[API CHAT] Set already_ingested to {config['already_ingested']}")
        
        # Add context about files to the config
        config["context"] = {
            "last_file": last_message.get("content"),
            "has_files": has_previous_upload
        }
        
        logger.info(f"[API CHAT] Processing chat with config: {config}")
        response = await process_chat(
            messages=request.messages,
            config=config
        )
        logger.info(f"[API CHAT] Got response: {response}")
        
        if response and response.get("messages"):
            last_message = response["messages"][-1]
            return {"response": last_message['content']}
        else:
            return {"response": "I apologize, but I couldn't generate a proper response."}
            
    except Exception as e:
        logger.error(f"[api.py] Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Mount static files for your frontend (adjust directory paths as needed)
app.mount("/static", StaticFiles(directory="static/static"), name="static")
app.mount("/", StaticFiles(directory="static", html=True), name="root")