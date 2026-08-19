import unittest
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
        collection.find_one_and_update.assert_awaited_with(
            {"thread_id": "thread"},
            {"$setOnInsert": {"thread_id": "thread", "user_id": "user-2"}},
            projection={"_id": 0, "user_id": 1},
            return_document=1,
            upsert=True,
        )
