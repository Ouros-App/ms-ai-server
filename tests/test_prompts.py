import unittest

from app.agents.prompts import SPECIALIST_JSON_RULES, SYSTEM_PROMPT


class PromptTest(unittest.TestCase):
    def test_default_prompt_is_the_only_natural_language_agent(self) -> None:
        self.assertIn("unico agente que conversa diretamente", SYSTEM_PROMPT)
        self.assertIn("Nao consulte tools, MCP ou memoria", SYSTEM_PROMPT)

    def test_specialists_have_a_json_contract(self) -> None:
        self.assertIn('"status": "ok|needs_input|unsupported|error"', SPECIALIST_JSON_RULES)
        self.assertIn('"missing_data"', SPECIALIST_JSON_RULES)
