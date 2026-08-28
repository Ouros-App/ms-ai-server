import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jwt

from app.agents.mcp import MCPToolProvider


class MCPProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_provider_filters_tools_and_issues_user_jwt(self) -> None:
        captured = {}
        captured["calls"] = 0

        class FakeClient:
            def __init__(self, connections, **kwargs):
                captured["connections"] = connections
                captured["kwargs"] = kwargs

            async def get_tools(self, server_name):
                captured["calls"] += 1
                self.server_name = server_name
                return [
                    SimpleNamespace(
                        name="get_user_farm_data",
                        description="Dados da fazenda",
                    ),
                    SimpleNamespace(name="search_knowledge", description="Busca"),
                    SimpleNamespace(name="unexpected_tool", description="Extra"),
                ]

        secret = "s" * 32
        provider = MCPToolProvider(
            url="http://mcp.test/mcp",
            jwt_secret=secret,
            issuer_url="https://issuer.test",
            resource_url="http://mcp.test/mcp",
        )

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            FakeClient,
        ):
            tools = await provider.tools_for("ranking", "42")
            cached_tools = await provider.tools_for("faq", "42")

        self.assertEqual([tool.name for tool in tools], ["get_user_farm_data"])
        self.assertEqual([tool.name for tool in cached_tools], ["search_knowledge"])
        self.assertEqual(captured["calls"], 1)
        connection = captured["connections"]["midas"]
        token = connection["headers"]["Authorization"].removeprefix("Bearer ")
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="http://mcp.test/mcp",
            issuer="https://issuer.test",
        )
        self.assertEqual(claims["sub"], "42")
        self.assertEqual(claims["user_type"], "farm_owner")

    async def test_provider_binds_user_identity_to_user_tools(self) -> None:
        remote_tool = SimpleNamespace(
            name="get_user_context",
            description="Contexto do usuario",
            ainvoke=AsyncMock(return_value="context"),
        )

        class FakeClient:
            def __init__(self, connections, **kwargs):
                pass

            async def get_tools(self, server_name):
                return [remote_tool]

        provider = MCPToolProvider(
            url="http://mcp.test/mcp",
            access_token="token",
            user_type="farm_owner",
        )

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            FakeClient,
        ):
            tools = await provider.tools_for("ranking", "42")

        self.assertEqual([tool.name for tool in tools], ["get_user_context"])
        self.assertEqual(await tools[0].ainvoke({}), "context")
        remote_tool.ainvoke.assert_awaited_once_with(
            {"user_type": "farm_owner", "user_id": 42}
        )

    async def test_farm_data_tool_filters_to_authorized_farm(self) -> None:
        remote_tool = SimpleNamespace(
            name="get_user_farm_data",
            description="Dados da fazenda",
            ainvoke=AsyncMock(
                return_value={
                    "user_type": "farm_owner",
                    "user_id": 42,
                    "farm_ids": [11],
                    "data": {
                        "farms": [{"id": 11, "name": "Fazenda autorizada"}],
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

        provider = MCPToolProvider(
            url="http://mcp.test/mcp",
            access_token="token",
        )

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            FakeClient,
        ):
            tools = await provider.tools_for("ranking", "42")
            authorized = await tools[0].ainvoke({"farm_id": 11})
            denied = await tools[0].ainvoke({"farm_id": 99})
            missing = await tools[0].ainvoke({})

        self.assertTrue(authorized["authorized"])
        self.assertEqual(authorized["data"]["farms"], [{"id": 11, "name": "Fazenda autorizada"}])
        self.assertEqual(authorized["data"]["water_registries"], [{"id_farm": 11, "value": 5}])
        self.assertFalse(denied["authorized"])
        self.assertEqual(denied["data"], {})
        self.assertEqual(missing, {"user_id": 42, "authorized": False, "data": {}})

    async def test_provider_skips_mcp_for_non_numeric_user(self) -> None:
        provider = MCPToolProvider(
            url="http://mcp.test/mcp",
            jwt_secret="s" * 32,
        )

        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client:
            tools = await provider.tools_for("ranking", "user-1")

        self.assertEqual(tools, [])
        client.assert_not_called()

    async def test_default_agent_has_no_mcp_allowlist(self) -> None:
        provider = MCPToolProvider(
            url="http://mcp.test/mcp",
            access_token="token",
        )

        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client:
            tools = await provider.tools_for("default", "42")

        self.assertEqual(tools, [])
        client.assert_not_called()
