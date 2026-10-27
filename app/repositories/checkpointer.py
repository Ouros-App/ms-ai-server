from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from app.core.config import settings


@contextmanager
def get_checkpointer() -> Iterator[MongoDBSaver]:
    """Abre o checkpointer MongoDB configurado para o servico."""
    client = MongoClient(settings.mongodb_uri)
    try:
        # shortcut: bypasses dependency setup; replace if a no-index constructor is added.
        checkpointer = MongoDBSaver.__new__(MongoDBSaver)
        BaseCheckpointSaver.__init__(checkpointer)
        checkpointer.client = client
        checkpointer.db = client[settings.mongodb_database]
        checkpointer.checkpoint_collection = checkpointer.db["checkpoints"]
        checkpointer.writes_collection = checkpointer.db["checkpoint_writes"]
        checkpointer.ttl = None
        yield checkpointer
    finally:
        client.close()
