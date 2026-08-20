import unittest

from app.agents.prompts import SYSTEM_PROMPT


class PromptTest(unittest.TestCase):
    def test_system_prompt_defines_personalized_memory_workflow(self) -> None:
        self.assertIn("recall_user_memories", SYSTEM_PROMPT)
        self.assertIn("save_user_memory", SYSTEM_PROMPT)
        self.assertIn("mesmo que ele nao use literalmente a palavra", SYSTEM_PROMPT)
        self.assertIn("A mensagem atual sempre", SYSTEM_PROMPT)
        self.assertIn("Nao transforme toda mensagem da conversa em memoria", SYSTEM_PROMPT)
