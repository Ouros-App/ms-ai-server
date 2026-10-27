import unittest
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.errors import OperationFailure

import app.repositories.checkpointer as checkpointer_module


class CheckpointerTest(unittest.TestCase):
    def test_get_checkpointer_does_not_inspect_or_create_indexes(self) -> None:
        client = MagicMock()
        database = MagicMock()
        collections = {
            name: MagicMock()
            for name in ("checkpoints", "checkpoint_writes")
        }
        for collection in collections.values():
            collection.list_indexes.side_effect = OperationFailure("not authorized", 8000)
            collection.create_index.side_effect = OperationFailure("not authorized", 8000)
        database.__getitem__.side_effect = collections.__getitem__
        client.__getitem__.return_value = database

        with (
            patch.object(checkpointer_module, "MongoClient", return_value=client, create=True),
            patch.object(
                MongoDBSaver,
                "from_conn_string",
                side_effect=OperationFailure("listIndexes not authorized", 8000),
            ) as from_conn_string,
            checkpointer_module.get_checkpointer() as checkpointer,
        ):
            self.assertIs(checkpointer.client, client)
            self.assertIs(checkpointer.checkpoint_collection, collections["checkpoints"])
            self.assertIs(
                checkpointer.writes_collection,
                collections["checkpoint_writes"],
            )

        from_conn_string.assert_not_called()
        client.close.assert_called_once()
        for collection in collections.values():
            collection.list_indexes.assert_not_called()
            collection.create_index.assert_not_called()
