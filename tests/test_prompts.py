import unittest

from app.agents.prompts import (
    COMMON_AGENT_RULES,
    MEMORY_AGENT_RULES,
    RANKING_AGENT_PROMPT,
    SPECIALIST_JSON_RULES,
    SYSTEM_PROMPT,
)


class PromptTest(unittest.TestCase):
    def test_default_prompt_is_the_only_natural_language_agent(self) -> None:
        self.assertIn("unico agente que conversa diretamente", SYSTEM_PROMPT)
        self.assertIn("Nao consulte tools, MCP ou memoria", SYSTEM_PROMPT)

    def test_specialists_have_a_json_contract(self) -> None:
        self.assertIn('"status": "ok|needs_input|unsupported|error"', SPECIALIST_JSON_RULES)
        self.assertIn('"missing_data"', SPECIALIST_JSON_RULES)

    def test_memory_rules_are_only_in_specialist_prompts(self) -> None:
        self.assertIn("recall_user_memories", MEMORY_AGENT_RULES)
        self.assertIn("recall_user_memories", RANKING_AGENT_PROMPT)
        self.assertNotIn("recall_user_memories", SYSTEM_PROMPT)

    def test_prompts_never_request_internal_identity_fields(self) -> None:
        self.assertIn("Nunca peca `farm_id`", COMMON_AGENT_RULES)
        self.assertIn("Nunca peca identificadores internos", SYSTEM_PROMPT)


    def test_product_rules_are_loaded_from_authorized_knowledge(self) -> None:
        self.assertIn("base de conhecimento", COMMON_AGENT_RULES)
        self.assertIn("search_knowledge", RANKING_AGENT_PROMPT)
        self.assertNotIn("biblioteca Explorar", COMMON_AGENT_RULES)
        self.assertNotIn("niveis ferro, bronze, prata e ouro", RANKING_AGENT_PROMPT)
