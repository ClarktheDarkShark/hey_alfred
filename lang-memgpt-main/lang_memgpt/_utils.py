from __future__ import annotations

from functools import lru_cache
import uuid

import langsmith
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
# from langchain_fireworks import FireworksEmbeddings
from pinecone import Pinecone

from lang_memgpt import _schemas as schemas
from lang_memgpt import _settings as settings

_DEFAULT_DELAY = 60  # seconds


def get_index():
    pc = Pinecone(api_key=settings.SETTINGS.pinecone_api_key)
    return pc.Index(settings.SETTINGS.pinecone_index_name)


@langsmith.traceable
def ensure_configurable(config: RunnableConfig) -> schemas.GraphConfig:
    """Merge the user-provided config with default values."""
    configurable = config.get("configurable", {})
    if not configurable:
        configurable = {
            "model": "gpt-4o-mini",
            "user_id": "default_user",
            "thread_id": str(uuid.uuid4())
        }
    elif "thread_id" not in configurable:
        configurable["thread_id"] = str(uuid.uuid4())
    return {
        **configurable,
        **schemas.GraphConfig(
            delay=configurable.get("delay", _DEFAULT_DELAY),
            model=configurable.get("model", settings.SETTINGS.model),
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        ),
    }


@lru_cache
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")


__all__ = ["ensure_configurable"]
