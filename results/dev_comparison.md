# So sánh trên dev (Day 7)

## Retrieval trên câu hỏi gốc (không qua agent), top-3, budget 2600 ký tự

| Phương pháp | Filter | Hit@1 | Evidence Hit@3 | MRR@3 | Context Evidence | Median ms/query |
|---|---|---|---|---|---|---|
| dense | có | 15/17 (88%) | 17/17 (100%) | 0.941 | 17/17 (100%) | 132.93 |
| bm25 | có | 14/17 (82%) | 16/17 (94%) | 0.873 | 16/17 (94%) | 0.43 |
| bm25_bigram | có | 14/17 (82%) | 16/17 (94%) | 0.882 | 16/17 (94%) | 0.51 |
| hybrid | có | 15/17 (88%) | 17/17 (100%) | 0.931 | 17/17 (100%) | 138.55 |
| hybrid_bigram | có | 15/17 (88%) | 17/17 (100%) | 0.941 | 17/17 (100%) | 151.16 |
| dense_nofilter | không | 14/17 (82%) | 17/17 (100%) | 0.902 | 17/17 (100%) | 139.27 |
| bm25_nofilter | không | 13/17 (76%) | 16/17 (94%) | 0.833 | 16/17 (94%) | 0.43 |
| bm25_bigram_nofilter | không | 13/17 (76%) | 16/17 (94%) | 0.843 | 16/17 (94%) | 0.56 |
| hybrid_nofilter | không | 15/17 (88%) | 17/17 (100%) | 0.931 | 17/17 (100%) | 140.85 |
| hybrid_bigram_nofilter | không | 15/17 (88%) | 17/17 (100%) | 0.941 | 17/17 (100%) | 145.1 |

## Agent (qwen3:4b-instruct, Ollama) trên dev: 17 câu có đáp án + 2 câu thiếu evidence

| Cấu hình | Retriever, top-k, budget, policy, thêm | Hit@1 / Hit@3 lượt tìm 1 | Context Evidence | Grounded 0/1/2 (TB /2) | Citation hợp lệ | Từ chối đúng | Từ chối nhầm | Lượt tìm TB | LLM calls TB | Median s/câu | Prompt tok TB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_dense | dense, k=3, 2600 ký tự, append | 13/17 (76%) / 14/17 (82%) | 14/17 (82%) | 3/1/13 (1.59) | 15/19 (79%) | 2/2 (100%) | 1/17 (6%) | 1 | 2.47 | 15.31 | 1903.2 |
| A1_bm25 | bm25_bigram, k=3, 2600 ký tự, append | 12/17 (71%) / 15/17 (88%) | 15/17 (88%) | 2/2/13 (1.65) | 15/16 (94%) | 2/2 (100%) | 1/17 (6%) | 1 | 2.47 | 13.83 | 1892.9 |
| A2_hybrid | hybrid_bigram, k=3, 2600 ký tự, append | 14/17 (82%) / 16/17 (94%) | 16/17 (94%) | 1/1/15 (1.82) | 16/17 (94%) | 2/2 (100%) | 0/17 (0%) | 1 | 2.47 | 14.99 | 1900.9 |
| A3_hybrid_orig | hybrid_bigram, k=3, 2600 ký tự, append, original_first | 15/17 (88%) / 17/17 (100%) | 17/17 (100%) | 0/0/17 (2.00) | 18/18 (100%) | 2/2 (100%) | 0/17 (0%) | 1 | 1 | 8.32 | 1228.3 |
| A4_hybrid_orig_ctx | hybrid_bigram, k=3, 4500 ký tự, best_rank, original_first, neighbors, research_prompt | 15/17 (88%) / 17/17 (100%) | 17/17 (100%) | 0/0/17 (2.00) | 18/18 (100%) | 2/2 (100%) | 0/17 (0%) | 1 | 1 | 10.6 | 1805.9 |

**Tiêu chí chọn:**

```
(applied in this order; written before any Day 7 test run):
  1. higher mean grounded-answer grade (/2, manual labels) on dev answerable questions
  2. fewer wrong abstentions + more correct abstentions on dev
  3. higher valid-citation rate (manual citation_support)
  4. higher Context Evidence Rate (gold evidence in the final context)
  5. lower median latency per question, then fewer prompt tokens (cost proxy; $0 local)
```

**Thứ hạng:** A3_hybrid_orig > A4_hybrid_orig_ctx > A2_hybrid > A1_bm25 > A0_dense

**Chọn:** `A3_hybrid_orig` (hoà 4 tiêu chí chất lượng với A4_hybrid_orig_ctx, thắng ở thời gian/token); baseline test: `A0_dense`.

Chi phí dịch vụ = $0 (embedding và LLM chạy local); thời gian đo trên laptop RTX 3050 Ti 4 GB, model ~77% GPU.
