import logging
import os
from typing import Optional

from llama_index.core.indices import load_index_from_storage
from llama_index.server.api.models import ChatRequest
from llama_index.server.tools.index.utils import get_storage_context
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

logger = logging.getLogger("uvicorn")

STORAGE_DIR = "storage"


def get_index():
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        return index
    except Exception as e:
        raise Exception(f"Failed to load index: {e}")
