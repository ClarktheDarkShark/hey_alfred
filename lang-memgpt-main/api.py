# lang-memgpt-main/api.py

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
from lang_memgpt.graph import process_chat
from lang_memgpt import _schemas as schemas
import os
import sys

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to capture all logs
    stream=sys.stdout,    # Ensure logs are sent to stdout
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    try:
        # Import modules only when needed
        from lang_memgpt.graph import detect_and_fix_docstring_issues
        detect_and_fix_docstring_issues()
    except Exception as e:
        logger.error(f"Non-critical error during docstring validation: {e}")
        # Continue app startup even if docstring validation fails

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://alfred-demo-311fd5c8f0bf.herokuapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    configurable: Dict[str, Any]

# First mount API endpoints
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: messages={request.messages} configurable={request.configurable}")
        print(f"Received chat request: messages={request.messages} configurable={request.configurable}", flush=True)
        # Create config as a regular dict
        config: schemas.GraphConfig = {
            "configurable": {
                "user_id": request.configurable.get("user_id", "default-user"),
                "model": request.configurable.get("model", "gpt-4o")
            }
        }
        
        response = await process_chat(
            messages=request.messages,
            config=config
        )
        
        # print(flush=True)
        # print(f'response={response}', flush=True)
        # print(flush=True)
        if response and response.get("messages"):
            last_message = response["messages"][-1]
            # print(f'Returning response: {last_message}', flush=True)
            return {"response": last_message['content']}
        else:
            return {"response": "I apologize, but I couldn't generate a proper response."}
            
    except Exception as e:
        logger.error(f"[api.py] Error in chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Then mount static files
app.mount("/static", StaticFiles(directory="static/static"), name="static")

# Finally mount the catch-all root
app.mount("/", StaticFiles(directory="static", html=True), name="root")
