import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

from app.repositories.thread_ownership import ThreadOwnershipStore


class ThreadOwnershipTest(unittest.IsolatedAsyncioTestCase):
    async def test_claim_uses_atomic_upsert_and_returns_owner_match(self) -> None:
        collection = Mock()
        collection.find_one_and_update = AsyncMock(return_value={"user_id": "user-1"})
        store = ThreadOwnershipStore({"thread_owners": collection})

        self.assertTrue(await store.claim("thread", "user-1"))
        self.assertFalse(await store.claim("thread", "user-2"))
        self.assertEqual(collection.find_one_and_update.await_count, 2)
        update = collection.find_one_and_update.await_args.args[1]
        self.assertEqual(
            update["$setOnInsert"],
            {
                "thread_id": "thread",
                "user_id": "user-2",
                "created_at": update["$setOnInsert"]["created_at"],
            },
        )
        self.assertIsInstance(update["$setOnInsert"]["created_at"], datetime)
        self.assertEqual(update["$set"]["last_activity_at"].tzinfo, timezone.utc)
        collection.find_one_and_update.assert_awaited_with(
            {"thread_id": "thread"},
            {
                "$setOnInsert": update["$setOnInsert"],
                "$set": update["$set"],
            },
            projection={"_id": 0, "user_id": 1},
            return_document=1,
            upsert=True,
        )

    async def test_set_title_only_when_missing_for_the_owner(self) -> None:
        collection = Mock()
        collection.update_one = AsyncMock(return_value=Mock(modified_count=1))
        store = ThreadOwnershipStore({"thread_owners": collection})

        self.assertTrue(
            await store.set_title_if_missing("thread", "user-1", "  Água   mensal ")
        )

        collection.update_one.assert_awaited_once_with(
            {
                "thread_id": "thread",
                "user_id": "user-1",
                "title": {"$exists": False},
            },
            {"$set": {"title": "Água mensal"}},
        )

    async def test_set_title_rejects_blank_value(self) -> None:
        collection = Mock()
        collection.update_one = AsyncMock()
        store = ThreadOwnershipStore({"thread_owners": collection})

        self.assertFalse(await store.set_title_if_missing("thread", "user-1", "  "))

        collection.update_one.assert_not_awaited()

    async def test_list_for_user_returns_recent_owned_threads(self) -> None:
        class Cursor:
            def sort(self, fields):
                self.sort_fields = fields
                return self

            def limit(self, count):
                self.limit_count = count
                return self

            def __aiter__(self):
                async def values():
                    yield {"thread_id": "thread-1", "title": "Título"}

                return values()

        collection = Mock()
        collection.find = Mock(return_value=Cursor())
        store = ThreadOwnershipStore({"thread_owners": collection})

        threads = await store.list_for_user("user-1", limit=40)

        self.assertEqual(threads, [{"thread_id": "thread-1", "title": "Título"}])
        collection.find.assert_called_once_with(
            {"user_id": "user-1"},
            projection={
                "_id": 0,
                "thread_id": 1,
                "title": 1,
                "last_activity_at": 1,
            },
        )
