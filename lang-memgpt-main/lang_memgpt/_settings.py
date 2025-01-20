from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
from typing import Literal

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(__file__), ".env"))  # Dynamic path resolution


class Settings(BaseSettings):
    # API Keys for embeddings and tools
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")


SETTINGS = Settings()
