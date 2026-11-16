import unittest

from advanced.agent import AgentConfig, calculate, parse_citations, run_agent
from advanced.run_eval import page_hits


class Calc(unittest.TestCase):
    def test_arithmetic_with_separators(self):
        self.assertTrue(calculate("9,583,686 - 9,451,106").endswith("= 132,580"))
        self.assertIn("4.0075", calculate("1182608 / 29509548 * 100"))

    def test_rejects_code(self):
        self.assertTrue(calculate("__import__('os').getcwd()").startswith("error"))


class Pages(unittest.TestCase):
    def test_required_groups(self):
        q = {"accept_pages": [247, 31], "required_pages": [[247], [31, 247]]}
        self.assertTrue(page_hits(q, [100, 247])["all_required"])
        self.assertEqual(page_hits(q, [100, 247])["first_hit_rank"], 2)
        q2 = {"accept_pages": [12, 6], "required_pages": [[12], [6, 8]]}
        self.assertFalse(page_hits(q2, [12, 31])["all_required"])


class FakeRetriever:
    name = "fake"

    def search(self, query, *, filters=None, top_k=4):
        return [{"id": "c1", "text": "Net interest income | 10 | 8", "page": 31, "rank": 1, "score": 1.0,
                 "metadata": {"kind": "table_text", "printed_page": 29}}]


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None):
        r = self.replies.pop(0)
        return {"content": r.get("content", ""), "tool_calls": r.get("tool_calls", []),
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "wall_s": 0}}


class AgentLoop(unittest.TestCase):
    def test_calculate_then_answer(self):
        llm = FakeLLM([{"tool_calls": [{"name": "calculate", "arguments": {"expression": "10-8"}}]},
                       {"content": "Up by 2 [c1]."}])
        tr = run_agent("q", llm, FakeRetriever(), AgentConfig())
        self.assertEqual(tr["status"], "answered")
        self.assertEqual(tr["calculations"][0]["result"], "10-8 = 2")
        self.assertEqual(parse_citations(tr["answer"]), ["c1"])

    def test_citation_prefixes_stripped(self):
        self.assertEqual(parse_citations("x [chunk-id: c1] y [Citation: c2 | page 3]"), ["c1", "c2"])

    def test_invalid_citation_abstains(self):
        llm = FakeLLM([{"content": "It is 10 [zzz]."}])
        self.assertEqual(run_agent("q", llm, FakeRetriever(), AgentConfig())["status"], "abstain")


if __name__ == "__main__":
    unittest.main()
