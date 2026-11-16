"""Offline checks for src/ (BM25, RRF, metrics, context, agent loop).

Fake LLM/retriever objects only test control flow; they never produce reported numbers.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import context, metrics  # noqa: E402
from src.agent import AgentConfig, parse_citations, run_agent  # noqa: E402
from src.retrievers import BM25Retriever, HybridRetriever, tokenize  # noqa: E402


def chunk(cid, text, page=1, section="s1"):
    return {"id": cid, "text": text, "source_id": "doc", "source": "doc.pdf", "page": page,
            "metadata": {"section": section}}


CHUNKS = [chunk("c1", "Doanh thu thuần năm 2022 đạt 258,880 triệu VND", 5, "tong_quan"),
          chunk("c2", "Có 4 loại tồn kho bao gồm nguyên vật liệu", 19, "ton_kho"),
          chunk("c3", "Quy trình quản lý rủi ro gồm phân tích rủi ro", 24, "rui_ro")]


class FakeRetriever:
    def __init__(self, name, ranking):
        self.name, self.ranking, self.calls = name, ranking, []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        by_id = {c["id"]: c for c in CHUNKS}
        return [{**by_id[i], "score": 1 / r, "rank": r, "strategy": self.name}
                for r, i in enumerate(self.ranking[:top_k], 1)]


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.seen = iter(replies), []

    def chat(self, messages, tools=None):
        self.seen.append((messages, tools))
        content, calls = next(self.replies)
        return {"content": content, "tool_calls": calls,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "wall_s": 0.0}}


def search_call(q):
    return [{"name": "search_documents", "arguments": {"query": q}}]


class RetrievalTests(unittest.TestCase):
    def test_tokenize_bigrams(self):
        self.assertEqual(tokenize("Tồn kho"), ["tồn", "kho"])
        self.assertIn("tồn_kho", tokenize("Tồn kho", bigrams=True))

    def test_bm25_ranks_and_filters(self):
        bm25 = BM25Retriever(CHUNKS)
        self.assertEqual(bm25.search("loại tồn kho", top_k=3)[0]["id"], "c2")
        self.assertEqual(bm25.search("tồn kho", filters={"section": "rui_ro"}, top_k=3), [])
        with self.assertRaises(ValueError):
            bm25.search("x", filters={"page": 1})

    def test_rrf_fuses_by_rank(self):
        hybrid = HybridRetriever([FakeRetriever("a", ["c1", "c2"]), FakeRetriever("b", ["c2", "c3"])], k=60)
        out = hybrid.search("q", top_k=3)
        self.assertEqual(out[0]["id"], "c2")  # ranked by both lists
        self.assertEqual([r["rank"] for r in out], [1, 2, 3])
        self.assertEqual(set(out[0]["components"]), {"a", "b"})


class MetricTests(unittest.TestCase):
    Q_ANY = {"gold": [{"source_id": "doc", "page": 19, "evidence": "4 loại  tồn kho"}], "gold_mode": "any"}
    Q_ALL = {"gold": [{"source_id": "doc", "page": 5, "evidence": "doanh thu thuần"},
                      {"source_id": "doc", "page": 19, "evidence": "4 loại tồn kho"}], "gold_mode": "all"}

    def test_hit_is_whitespace_insensitive_and_page_bound(self):
        self.assertTrue(metrics.is_hit(CHUNKS[1], self.Q_ANY))
        self.assertFalse(metrics.is_hit({**CHUNKS[1], "page": 20}, self.Q_ANY))

    def test_all_mode_needs_every_evidence(self):
        self.assertEqual(metrics.first_cover_rank([CHUNKS[0], CHUNKS[2], CHUNKS[1]], self.Q_ALL), 3)
        self.assertIsNone(metrics.first_cover_rank([CHUNKS[0]], self.Q_ALL))


class ContextTests(unittest.TestCase):
    def test_best_rank_prefers_later_top_hit(self):
        cands = [{**CHUNKS[0], "rank": 2}, {**CHUNKS[1], "rank": 1}]
        self.assertEqual(context.order(cands, "append")[0]["id"], "c1")
        self.assertEqual(context.order(cands, "best_rank")[0]["id"], "c2")

    def test_neighbor_added_after_parent_without_displacing_hits(self):
        nxt = {"c1": CHUNKS[1], "c2": CHUNKS[2]}
        _, ids, kept = context.pack([CHUNKS[0]], 10_000, "append", nxt)
        self.assertEqual(ids, ["c1", "c2"])
        self.assertEqual(kept[1]["neighbor_of"], "c1")
        _, ids, _ = context.pack([CHUNKS[0]], len(context.block(CHUNKS[0])), "append", nxt)
        self.assertEqual(ids, ["c1"])  # neighbour dropped when the budget is full

    def test_budget_skips_chunks_that_do_not_fit(self):
        one = len(context.block(CHUNKS[0]))
        text, ids, _ = context.pack(CHUNKS, one)
        self.assertEqual(ids, ["c1"])
        self.assertLessEqual(len(text), one)


class AgentTests(unittest.TestCase):
    def test_parse_citations(self):
        self.assertEqual(parse_citations("A [c1] B [c2, c3] C [c1]"), ["c1", "c2", "c3"])

    def test_search_then_answer_with_valid_citation(self):
        llm = FakeLLM([("", search_call("tồn kho")), ("Có 4 loại [c2].", [])])
        r = FakeRetriever("fake", ["c2", "c1"])
        out = run_agent("Có mấy loại tồn kho?", llm, r, AgentConfig(budget=5000), filters={"section": "ton_kho"})
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["c2"])
        self.assertEqual(r.calls, [("tồn kho", {"section": "ton_kho"}, 3)])

    def test_citation_outside_context_abstains(self):
        llm = FakeLLM([("", search_call("x")), ("Trả lời [c9].", [])])
        out = run_agent("q?", llm, FakeRetriever("fake", ["c1"]), AgentConfig(budget=5000))
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["events"][-1]["event"], "invalid_or_missing_citations")

    def test_explicit_abstention_and_search_limit(self):
        llm = FakeLLM([("", search_call("a")), ("", search_call("b")),
                       ("KHÔNG ĐỦ BẰNG CHỨNG: không có", [])])
        out = run_agent("q?", llm, FakeRetriever("fake", ["c1"]), AgentConfig(max_searches=2, budget=5000))
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["n_searches"], 2)
        self.assertIsNone(llm.seen[-1][1])  # no tools offered after the limit

    def test_original_first_searches_question_verbatim(self):
        llm = FakeLLM([("Doanh thu [c1]", [])])
        r = FakeRetriever("fake", ["c1"])
        out = run_agent("Doanh thu 2022?", llm, r, AgentConfig(budget=5000, original_first=True))
        self.assertEqual(r.calls[0][0], "Doanh thu 2022?")
        self.assertEqual(out["events"][0]["tool"], "system_original_query")
        self.assertEqual(out["status"], "answered")

    def test_answer_before_search_is_rejected_once(self):
        llm = FakeLLM([("Đáp án [c1]", []), ("", search_call("q")), ("Đáp án [c1]", [])])
        out = run_agent("q?", llm, FakeRetriever("fake", ["c1"]), AgentConfig(budget=5000))
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["events"][0]["event"], "answer_without_search_rejected")


if __name__ == "__main__":
    unittest.main()
