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
        """Replace token exchange with a deterministic delegated-token stub."""
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
                        name="get_user_context",
                        description="Contexto seguro",
                        ainvoke=AsyncMock(),
                    ),
                    SimpleNamespace(
                        name="get_user_farm_data",
                        description="Dados brutos",
                        ainvoke=AsyncMock(),
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
            ranking = await provider.tools_for("ranking")
            faq = await provider.tools_for("faq")

        self.assertEqual(
            [tool.name for tool in ranking],
            ["get_user_context", "search_knowledge"],
        )
        self.assertEqual(
            [tool.name for tool in faq],
            ["get_user_context", "search_knowledge"],
        )
        self.assertNotIn("get_user_farm_data", [tool.name for tool in ranking])
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
            "profile": {
                "id": 42,
                "name": "Produtor",
                "id_user": 42,
                "email": "private@example.com",
                "telephone": "5511999999999",
                "document_number": "00000000000",
            },
            "farms": [
                {
                    "id": 11,
                    "id_farm": 11,
                    "name": "Fazenda",
                    "area_property": 120,
                    "region": "Sudeste",
                    "poultry_capacity": 5000,
                    "place": "Zona rural",
                    "state": "SP",
                    "city": "Campinas",
                    "email": "farm-private@example.com",
                    "owner_name": "Nao deve vazar",
                }
            ],
            "enterprises": [
                {
                    "id": 7,
                    "name": "Empresa",
                    "email": "billing@example.com",
                    "telephone": "5511888888888",
                }
            ],
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
            tools = await provider.tools_for("faq")
            result = await tools[0].ainvoke({})
            self.assertEqual(
                result,
                {
                    "profile": {"name": "Produtor"},
                    "farms": [
                        {
                            "name": "Fazenda",
                            "area_property": 120,
                            "region": "Sudeste",
                            "poultry_capacity": 5000,
                            "place": "Zona rural",
                            "state": "SP",
                            "city": "Campinas",
                        }
                    ],
                    "enterprises": [{"name": "Empresa"}],
                },
            )

        call = remote_context.ainvoke.await_args.args[0]
        self.assertEqual(call["type"], "tool_call")
        self.assertEqual(call["name"], "get_user_context")
        self.assertEqual(call["args"], {})
        self.assertNotIn("user_id", call["args"])
        self.assertNotIn("private@example.com", str(result))
        self.assertNotIn("5511999999999", str(result))
        self.assertNotIn("00000000000", str(result))
        self.assertNotIn("billing@example.com", str(result))
        self.assertNotIn("5511888888888", str(result))
        self.assertNotIn("farm-private@example.com", str(result))
        self.assertNotIn("Nao deve vazar", str(result))

    async def test_consumption_summary_rejects_mismatched_period(self) -> None:
        payload = {
            "user_type": "farm_owner",
            "user_id": 42,
            "farm_ids": [11],
            "period_days": 7,
            "water_unit": "hydrometer_reading_delta",
            "energy_unit": "kWh",
            "summaries": [],
        }

        with self.assertRaises(MCPToolResultError):
            MCPToolProvider._filter_consumption_summary(
                payload,
                expected_period_days=30,
            )

    async def test_consumption_summary_keeps_scope_and_only_exposes_period(self) -> None:
        payload = {
            "user_type": "farm_owner",
            "user_id": 42,
            "farm_ids": [11],
            "period_days": 30,
            "water_unit": "hydrometer_reading_delta",
            "energy_unit": "kWh",
            "summaries": [
                {"id_farm": 11, "water_meter_delta": 80},
                {"id_farm": 99, "water_meter_delta": 999},
            ],
        }
        remote_tool = SimpleNamespace(
            name="get_consumption_summary",
            description="Resumo de consumo",
            ainvoke=AsyncMock(
                return_value=ToolMessage(
                    content=[{"type": "text", "text": "ignored"}],
                    artifact={"structured_content": payload},
                    tool_call_id="remote-call",
                    name="get_consumption_summary",
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
            tools = await provider.tools_for("sustainability")
            result = await tools[0].ainvoke({"period_days": 30})

        self.assertEqual(set(tools[0].args_schema.model_fields), {"period_days"})
        self.assertTrue(result["authorized"])
        self.assertEqual(result["water_unit"], "hydrometer_reading_delta")
        self.assertEqual(result["energy_unit"], "kWh")
        self.assertEqual(
            result["summaries"],
            [{"water_meter_delta": 80}],
        )
        self.assertNotIn("user_id", result)
        self.assertNotIn("user_type", result)
        self.assertNotIn("id_farm", result["summaries"][0])
        call = remote_tool.ainvoke.await_args.args[0]
        self.assertEqual(call["args"], {"period_days": 30})
        self.assertNotIn("farm_id", call["args"])
        self.assertNotIn("user_id", call["args"])

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
        """Reject malformed user-scoped results instead of trusting partial data."""
        valid_context = {
            "user_type": "farm_owner",
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
        for payload in (
            {"user_type": "farm_owner"},
            {**valid_context, "profile": []},
        ):
            with self.subTest(payload=payload):
                self.assertFalse(
                    MCPToolProvider._result_contract_is_valid(
                        payload,
                        tool_name="get_user_context",
                    )
                )

        valid_summary = {
            "user_type": "farm_owner",
            "user_id": 42,
            "farm_ids": [11],
            "period_days": 30,
            "summaries": [],
        }
        self.assertTrue(
            MCPToolProvider._result_contract_is_valid(
                valid_summary,
                tool_name="get_consumption_summary",
                expected_period_days=30,
            )
        )
        for payload in (
            {"error": "timeout"},
            {**valid_summary, "farm_ids": "11"},
            {**valid_summary, "farm_ids": [True]},
            {**valid_summary, "summaries": {}},
            {**valid_summary, "period_days": 7},
        ):
            with self.subTest(payload=payload):
                self.assertFalse(
                    MCPToolProvider._result_contract_is_valid(
                        payload,
                        tool_name="get_consumption_summary",
                        expected_period_days=30,
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
                tool_name="get_consumption_summary",
                expected_period_days=30,
            )

        invalid = [
            event
            for event in events
            if event["event"] == "mcp.tool_result_invalid"
        ]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["tool"], "get_consumption_summary")
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

    def test_remote_tool_error_is_traced_with_bounded_message(self) -> None:
        """Surface MCP execution errors distinctly from malformed result payloads."""
        result = ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": "Error executing tool get_user_context: upstream unavailable",
                }
            ],
            artifact=None,
            tool_call_id="remote-error",
            name="get_user_context",
            status="error",
        )

        with (
            capture_debug_trace() as events,
            self.assertRaises(MCPToolResultError),
        ):
            MCPToolProvider._require_decoded_result(
                result,
                tool_name="get_user_context",
            )

        remote_errors = [
            event
            for event in events
            if event["event"] == "mcp.tool_error"
        ]
        self.assertEqual(len(remote_errors), 1)
        self.assertGreater(remote_errors[0]["error_chars"], 0)
        self.assertNotIn("message", remote_errors[0])
        self.assertNotIn("upstream unavailable", str(remote_errors[0]))
        self.assertFalse(
            any(event["event"] == "mcp.tool_result_invalid" for event in events)
        )

    def test_tool_message_error_text_is_bounded_and_has_fallback(self) -> None:
        """Keep remote error diagnostics concise and useful."""
        long_message = ToolMessage(
            content="x" * 800,
            tool_call_id="remote-error",
            name="get_user_context",
            status="error",
        )
        self.assertEqual(
            len(MCPToolProvider._tool_message_text(long_message)),
            500,
        )

        empty_message = ToolMessage(
            content=[],
            tool_call_id="remote-error",
            name="get_user_context",
            status="error",
        )
        self.assertEqual(
            MCPToolProvider._tool_message_text(empty_message),
            "MCP tool returned an unspecified error.",
        )

    async def test_invalid_mcp_result_is_not_reported_as_authorization_denial(
        self,
    ) -> None:
        """Treat undecodable MCP output as integration failure, not scope denial."""
        remote_tool = SimpleNamespace(
            name="get_user_context",
            description="Contexto",
            ainvoke=AsyncMock(
                return_value=ToolMessage(
                    content=[{"type": "text", "text": "not-json"}],
                    artifact=None,
                    tool_call_id="remote-call",
                    name="get_user_context",
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
            tools = await provider.tools_for("faq")
            with self.assertRaises(MCPToolResultError):
                await tools[0].ainvoke({})

        invalid = [
            event
            for event in events
            if event["event"] == "mcp.tool_result_invalid"
        ]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["tool"], "get_user_context")
        self.assertEqual(invalid[0]["result_type"], "ToolMessage")

    async def test_provider_exposes_no_mcp_tools_without_forwarded_user_token(self) -> None:
        """Expose no user-scoped MCP tools when no validated token was forwarded."""
        provider = MCPToolProvider(url="http://mcp.test/mcp")
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client:
            tools = await provider.tools_for("ranking")

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
            tools = await provider.tools_for("faq")

        self.assertEqual(tools, [])
        client.assert_not_called()
        failures = [
            event
            for event in events
            if event["event"] == "mcp.token_exchange_failed"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["agent"], "faq")
        self.assertEqual(failures[0]["reason"], "unknown")
        self.assertIsNone(failures[0]["status"])

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
            tools = await provider.tools_for("sustainability")

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
            tools = await provider.tools_for("default")

        self.assertEqual(tools, [])
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
