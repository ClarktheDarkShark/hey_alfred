# api.py
import os
import sys
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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
    print(f"[API CHAT] Received request with messages: {request.messages}", flush=True)
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
        print(f"[API CHAT] Set already_ingested to {config['already_ingested']}", flush=True)
        
        # Add context about files to the config
        config["context"] = {
            "last_file": last_message.get("content"),
            "has_files": has_previous_upload
        }
        
        print(f"[API CHAT] Processing chat with config: {config}", flush=True)
        response = await process_chat(
            messages=request.messages,
            config=config
        )
        print(f"[API CHAT] Got response: {response}", flush=True)
        
        if response and response.get("messages"):
            last_message = response["messages"][-1]
            return {"response": last_message['content']}
        else:
            return {"response": "I apologize, but I couldn't generate a proper response."}
            
    except Exception as e:
        logger.error(f"[api.py] Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"[API UPLOAD] Starting file upload process for: {file.filename}", flush=True)
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext == '.pdf':
            save_dir = PDF_DIR
        elif file_ext == '.csv':
            save_dir = CSV_DIR
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        safe_filename = os.path.basename(file.filename)
        file_path = os.path.join(save_dir, safe_filename)
        
        print(f"[API UPLOAD] Saving file to: {file_path}", flush=True)
        print(f"[API UPLOAD] Directory exists: {os.path.exists(os.path.dirname(file_path))}", flush=True)
        
        # Save the file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        print(f"[API UPLOAD] File saved, size: {len(contents)} bytes", flush=True)
        print(f"[API UPLOAD] File exists: {os.path.exists(file_path)}", flush=True)

        print(f"[API UPLOAD] Starting ingestion for: {safe_filename}", flush=True)
        ingest_result = await ingest_data.ainvoke({"filename": safe_filename})
        print(f"[API UPLOAD] Ingestion completed with result: {ingest_result}", flush=True)

        return JSONResponse(content={
            "filename": safe_filename,
            "status": "success",
            "message": "File uploaded and ingested successfully",
            "ingestion_result": ingest_result
        })

    except Exception as e:
        print(f"[API UPLOAD] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for your frontend (adjust directory paths as needed)
app.mount("/static", StaticFiles(directory="static/static"), name="static")
app.mount("/", StaticFiles(directory="static", html=True), name="root")
