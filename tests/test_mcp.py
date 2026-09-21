import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import ToolMessage

from app.agents.mcp import (
    MCP_TOOLS_CACHE_MAX_ENTRIES,
    MCPToolProvider,
    MCPToolResultError,
    forward_mcp_access_token,
)
from app.core.token_exchange import MCPTokenExchangeError
from app.debug_ui.trace import capture_debug_trace


class MCPProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.exchange_token = AsyncMock(
            side_effect=lambda token: f"delegated::{token}"
        )
        self.exchange_patcher = patch(
            "app.agents.mcp.exchange_mcp_access_token",
            new=self.exchange_token,
        )
        self.exchange_patcher.start()
        self.addCleanup(self.exchange_patcher.stop)

    def test_tools_cache_prunes_expired_and_bounds_entries(self) -> None:
        """Prune expired MCP tool cache entries and enforce the size limit."""
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
        """Forward the validated token while exposing only allowed MCP tools."""
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
            "Bearer delegated::signed-keycloak-token",
        )

    async def test_bound_user_tools_preserve_context_artifact_without_identity_args(
        self,
    ) -> None:
        """Preserve context artifacts without exposing identity arguments."""
        payload = {
            "user_type": "farm_owner",
            "user_id": 42,
            "profile": {"name": "Produtor"},
            "farms": [{"id": 11, "name": "Fazenda"}],
            "enterprises": [],
        }
        remote_context = SimpleNamespace(
            name="get_user_context",
            description="Contexto",
            ainvoke=AsyncMock(
                return_value=ToolMessage(
                    content=[{"type": "text", "text": "contexto estruturado"}],
                    artifact={"structured_content": payload},
                    tool_call_id="remote-call",
                    name="get_user_context",
                )
            ),
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
            self.assertEqual(await tools[0].ainvoke({}), payload)

        call = remote_context.ainvoke.await_args.args[0]
        self.assertEqual(call["type"], "tool_call")
        self.assertEqual(call["name"], "get_user_context")
        self.assertEqual(call["args"], {})
        self.assertNotIn("user_id", call["args"])

    async def test_farm_data_tool_decodes_structured_artifact_and_filters_scope(
        self,
    ) -> None:
        """Decode structured farm artifacts and filter rows to authorized farms."""
        payload = {
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
        remote_tool = SimpleNamespace(
            name="get_user_farm_data",
            description="Dados",
            ainvoke=AsyncMock(
                return_value=ToolMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "conteudo textual nao autoritativo",
                        }
                    ],
                    artifact={"structured_content": payload},
                    tool_call_id="remote-call",
                    name="get_user_farm_data",
                )
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
            result = await tools[0].ainvoke({"limit": 20})

        self.assertNotIn("farm_id", tools[0].args_schema.model_fields)
        self.assertTrue(result["authorized"])
        self.assertEqual(result["data"]["farms"], [{"id": 11}])
        self.assertEqual(
            result["data"]["water_registries"],
            [{"id_farm": 11, "value": 5}],
        )
        call = remote_tool.ainvoke.await_args.args[0]
        self.assertEqual(call["type"], "tool_call")
        self.assertEqual(call["name"], "get_user_farm_data")
        self.assertEqual(call["args"], {"limit": 20})

    def test_decode_tool_result_supports_adapter_artifact_shapes(self) -> None:
        """Decode the common content-and-artifact shapes emitted by the adapter."""
        payload = {
            "user_type": "farm_owner",
            "farm_ids": [11],
            "data": {"water_registries": [{"id_farm": 11}]},
        }

        self.assertEqual(
            MCPToolProvider._decode_tool_result(
                {"structured_content": payload}
            ),
            payload,
        )
        self.assertEqual(
            MCPToolProvider._decode_tool_result(
                ([{"type": "text", "text": "ignored"}], {"structured_content": payload})
            ),
            payload,
        )
        self.assertEqual(
            MCPToolProvider._decode_tool_result(
                [{"type": "text", "text": '{"farm_ids":[11],"data":{}}'}]
            ),
            {"farm_ids": [11], "data": {}},
        )

    def test_decoder_covers_nested_adapter_shapes_and_model_dump(self) -> None:
        """Decode structured MCP data from all supported adapter wrappers."""
        payload = {"farm_ids": [11], "data": {}}

        self.assertEqual(
            MCPToolProvider._decode_tool_result(
                {"structuredContent": payload}
            ),
            payload,
        )
        self.assertEqual(
            MCPToolProvider._decode_tool_result(
                {"artifact": {"structured_content": payload}}
            ),
            payload,
        )

        wrapped = SimpleNamespace(
            artifact={"structured_content": payload},
            content="ignored",
        )
        self.assertEqual(
            MCPToolProvider._decode_tool_result(wrapped),
            payload,
        )

        dumped = SimpleNamespace(
            artifact=None,
            content=None,
            model_dump=lambda: {"structured_content": payload},
        )
        self.assertEqual(
            MCPToolProvider._decode_tool_result(dumped),
            payload,
        )

        self.assertIsNone(
            MCPToolProvider._decode_tool_result("not-json")
        )

    def test_user_scoped_result_contracts_reject_error_payloads(self) -> None:
        """Reject malformed MCP dictionaries instead of treating them as scope state."""
        valid_farm = {"farm_ids": [11], "data": {}}
        self.assertTrue(
            MCPToolProvider._result_contract_is_valid(
                valid_farm,
                tool_name="get_user_farm_data",
            )
        )
        for payload in (
            {"error": "timeout"},
            {"farm_ids": "11", "data": {}},
            {"farm_ids": [True], "data": {}},
            {"farm_ids": [11], "data": []},
        ):
            with self.subTest(payload=payload):
                self.assertFalse(
                    MCPToolProvider._result_contract_is_valid(
                        payload,
                        tool_name="get_user_farm_data",
                    )
                )

        valid_context = {
            "user_type": "farm_owner",
            "user_id": 42,
            "profile": {},
            "farms": [],
            "enterprises": [],
        }
        self.assertTrue(
            MCPToolProvider._result_contract_is_valid(
                valid_context,
                tool_name="get_user_context",
            )
        )
        self.assertFalse(
            MCPToolProvider._result_contract_is_valid(
                {"user_type": "farm_owner", "user_id": 42},
                tool_name="get_user_context",
            )
        )
        self.assertFalse(
            MCPToolProvider._result_contract_is_valid(
                {**valid_context, "user_id": True},
                tool_name="get_user_context",
            )
        )

    def test_require_decoded_result_rejects_error_dict_and_traces_it(self) -> None:
        """Emit diagnostics for decoded dictionaries that violate the tool contract."""
        with (
            capture_debug_trace() as events,
            self.assertRaises(MCPToolResultError),
        ):
            MCPToolProvider._require_decoded_result(
                {"error": "timeout"},
                tool_name="get_user_farm_data",
            )

        invalid = [
            event
            for event in events
            if event["event"] == "mcp.tool_result_invalid"
        ]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["tool"], "get_user_farm_data")
        self.assertEqual(invalid[0]["result_type"], "dict")

    def test_decoder_skips_unsupported_blocks_before_valid_text(self) -> None:
        """Skip non-text MCP blocks and continue to a later JSON text result."""
        self.assertEqual(
            MCPToolProvider._decode_tool_result(
                [
                    {
                        "type": "image",
                        "base64": "ignored",
                        "mime_type": "image/png",
                    },
                    {
                        "type": "text",
                        "text": '{"farm_ids":[11],"data":{}}',
                    },
                ]
            ),
            {"farm_ids": [11], "data": {}},
        )

    async def test_invalid_mcp_result_is_not_reported_as_authorization_denial(
        self,
    ) -> None:
        """Treat undecodable MCP output as integration failure, not scope denial."""
        remote_tool = SimpleNamespace(
            name="get_user_farm_data",
            description="Dados",
            ainvoke=AsyncMock(
                return_value=ToolMessage(
                    content=[{"type": "text", "text": "not-json"}],
                    artifact=None,
                    tool_call_id="remote-call",
                    name="get_user_farm_data",
                )
            ),
        )

        class FakeClient:
            def __init__(self, connections, **kwargs):
                pass

            async def get_tools(self, server_name):
                return [remote_tool]

        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with (
            capture_debug_trace() as events,
            forward_mcp_access_token("signed-keycloak-token"),
            patch("langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient),
        ):
            tools = await provider.tools_for("sustainability", "42")
            with self.assertRaises(MCPToolResultError):
                await tools[0].ainvoke({"limit": 20})

        invalid = [
            event
            for event in events
            if event["event"] == "mcp.tool_result_invalid"
        ]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["tool"], "get_user_farm_data")
        self.assertEqual(invalid[0]["result_type"], "ToolMessage")

    async def test_empty_farm_scope_is_explicitly_distinct_from_decode_failure(
        self,
    ) -> None:
        """Represent a real empty farm scope separately from malformed MCP output."""
        result = MCPToolProvider._filter_farm_data(
            {
                "user_type": "company_employee",
                "user_id": 42,
                "farm_ids": [],
                "data": {},
            },
            42,
        )

        self.assertFalse(result["authorized"])
        self.assertEqual(result["reason"], "no_farm_scope")
        self.assertEqual(result["user_type"], "company_employee")

    async def test_provider_exposes_no_mcp_tools_without_forwarded_user_token(self) -> None:
        """Expose no user-scoped MCP tools when no validated token was forwarded."""
        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client:
            tools = await provider.tools_for("ranking", "42")

        self.assertEqual(tools, [])
        client.assert_not_called()

    async def test_token_exchange_failure_hides_tools_and_emits_trace(self) -> None:
        """Do not fall back to the user token when delegated exchange fails."""

        self.exchange_token.side_effect = MCPTokenExchangeError("exchange failed")
        provider = MCPToolProvider(url="http://mcp.test/mcp")

        with (
            capture_debug_trace() as events,
            forward_mcp_access_token("signed-keycloak-token"),
            patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client,
        ):
            tools = await provider.tools_for("faq", "42")

        self.assertEqual(tools, [])
        client.assert_not_called()
        failures = [
            event
            for event in events
            if event["event"] == "mcp.token_exchange_failed"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["agent"], "faq")
        self.assertEqual(failures[0]["error"], "MCPTokenExchangeError")

    async def test_mcp_load_failure_is_visible_in_debug_trace(self) -> None:
        """Expose MCP outages in diagnostics instead of silently hiding all tools."""

        class FailingClient:
            def __init__(self, connections, **kwargs):
                pass

            async def get_tools(self, server_name):
                raise RuntimeError("mcp unavailable")

        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with (
            capture_debug_trace() as events,
            forward_mcp_access_token("signed-keycloak-token"),
            patch(
                "langchain_mcp_adapters.client.MultiServerMCPClient",
                FailingClient,
            ),
        ):
            tools = await provider.tools_for("sustainability", "42")

        self.assertEqual(tools, [])
        failures = [
            event
            for event in events
            if event["event"] == "mcp.tools_load_failed"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["agent"], "sustainability")
        self.assertEqual(failures[0]["error"], "RuntimeError")

    async def test_default_agent_has_no_mcp_allowlist(self) -> None:
        """Keep MCP tools unavailable to the default synthesizer agent."""
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
