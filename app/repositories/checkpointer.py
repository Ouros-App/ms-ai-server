from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.mongodb import MongoDBSaver

from app.core.config import settings


@contextmanager
def get_checkpointer() -> Iterator[MongoDBSaver]:
    """Abre o checkpointer MongoDB configurado para o servico."""
    with MongoDBSaver.from_conn_string(
        settings.mongodb_uri,
        db_name=settings.mongodb_database,
    ) as checkpointer:
        yield checkpointer
