import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.agents.mcp import (
    MCP_TOOLS_CACHE_MAX_ENTRIES,
    MCPToolProvider,
    forward_mcp_access_token,
)


class MCPProviderTest(unittest.IsolatedAsyncioTestCase):
    def test_tools_cache_prunes_expired_and_bounds_entries(self) -> None:
        provider = MCPToolProvider(
            url="http://mcp.test/mcp",
            cache_ttl_seconds=10,
        )
        provider._tools_cache = {
            "expired": (["old"], 80.0),
            **{
                f"active-{index}": (["tool"], 95.0)
                for index in range(MCP_TOOLS_CACHE_MAX_ENTRIES)
            },
        }

        provider._prune_tools_cache(100.0)

        self.assertNotIn("expired", provider._tools_cache)
        self.assertLess(len(provider._tools_cache), MCP_TOOLS_CACHE_MAX_ENTRIES)
        self.assertNotIn("active-0", provider._tools_cache)

    async def test_provider_forwards_same_keycloak_token_and_filters_allowlist(self) -> None:
        captured = {"calls": 0}

        class FakeClient:
            def __init__(self, connections, **kwargs):
                captured["connections"] = connections
                captured["kwargs"] = kwargs

            async def get_tools(self, server_name):
                captured["calls"] += 1
                return [
                    SimpleNamespace(
                        name="get_user_farm_data",
                        description="Dados da fazenda",
                        ainvoke=AsyncMock(return_value={"farm_ids": [], "data": {}}),
                    ),
                    SimpleNamespace(
                        name="search_knowledge",
                        description="Busca",
                        ainvoke=AsyncMock(return_value=[]),
                    ),
                    SimpleNamespace(name="unexpected_tool", description="Extra"),
                ]

        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with (
            forward_mcp_access_token("signed-keycloak-token"),
            patch("langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient),
        ):
            ranking = await provider.tools_for("ranking", "42")
            faq = await provider.tools_for("faq", "42")

        self.assertEqual([tool.name for tool in ranking], ["get_user_farm_data"])
        self.assertEqual([tool.name for tool in faq], ["search_knowledge"])
        self.assertEqual(captured["calls"], 1)
        self.assertEqual(
            captured["connections"]["midas"]["headers"]["Authorization"],
            "Bearer signed-keycloak-token",
        )

    async def test_bound_user_tools_send_no_identity_arguments(self) -> None:
        remote_context = SimpleNamespace(
            name="get_user_context",
            description="Contexto",
            ainvoke=AsyncMock(return_value="context"),
        )

        class FakeClient:
            def __init__(self, connections, **kwargs):
                pass

            async def get_tools(self, server_name):
                return [remote_context]

        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with (
            forward_mcp_access_token("signed-keycloak-token"),
            patch("langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient),
        ):
            tools = await provider.tools_for("faq", "42")
            self.assertEqual(await tools[0].ainvoke({}), "context")

        remote_context.ainvoke.assert_awaited_once_with({})

    async def test_farm_data_filter_still_limits_returned_farms(self) -> None:
        remote_tool = SimpleNamespace(
            name="get_user_farm_data",
            description="Dados",
            ainvoke=AsyncMock(
                return_value={
                    "user_type": "farm_owner",
                    "user_id": 42,
                    "farm_ids": [11],
                    "data": {
                        "farms": [{"id": 11}, {"id": 99}],
                        "water_registries": [
                            {"id_farm": 11, "value": 5},
                            {"id_farm": 99, "value": 999},
                        ],
                    },
                }
            ),
        )

        class FakeClient:
            def __init__(self, connections, **kwargs):
                pass

            async def get_tools(self, server_name):
                return [remote_tool]

        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with (
            forward_mcp_access_token("signed-keycloak-token"),
            patch("langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient),
        ):
            tools = await provider.tools_for("ranking", "42")
            authorized = await tools[0].ainvoke({"farm_id": 11})
            denied = await tools[0].ainvoke({"farm_id": 99})

        self.assertTrue(authorized["authorized"])
        self.assertEqual(authorized["data"]["farms"], [{"id": 11}])
        self.assertFalse(denied["authorized"])

    async def test_provider_exposes_no_mcp_tools_without_forwarded_user_token(self) -> None:
        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client:
            tools = await provider.tools_for("ranking", "42")

        self.assertEqual(tools, [])
        client.assert_not_called()

    async def test_default_agent_has_no_mcp_allowlist(self) -> None:
        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with (
            forward_mcp_access_token("signed-keycloak-token"),
            patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client,
        ):
            tools = await provider.tools_for("default", "42")

        self.assertEqual(tools, [])
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
