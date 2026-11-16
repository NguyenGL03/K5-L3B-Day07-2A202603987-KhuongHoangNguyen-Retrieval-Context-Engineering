# Code Day 7 — Agentic RAG trên index Day 6

| File | Vai trò |
|---|---|
| `config.py` | Đường dẫn, kết nối pgvector Day 6, embedding model/revision, model Ollama |
| `prepare_data.py` | Xuất 91 chunk index B từ pgvector → `data/day6/chunks.json`; chuyển benchmark Day 6 + 4 câu thiếu evidence → `data/benchmark/` |
| `embedding.py` | Query encoder `google/embeddinggemma-2` (rev `f9972ac`, 768d, prompt query của Day 6) |
| `retrievers.py` | `dense` (pgvector), `bm25` / `bm25_bigram` (lexical), `hybrid` (RRF dense + BM25) |
| `context.py` | Dedupe, thứ tự context (`append` / `best_rank`), budget theo ký tự |
| `metrics.py` | Định nghĩa hit (nguồn + trang + evidence, bỏ khác biệt khoảng trắng), Hit@1, Hit@3, MRR, Context Evidence |
| `retrieval_eval.py` | Benchmark retrieval trên câu hỏi gốc, không qua agent |
| `llm.py`, `agent.py` | Client Ollama (tool calling) và vòng lặp agent: search → context → trả lời có `[chunk-id]` hoặc `KHÔNG ĐỦ BẰNG CHỨNG` |
| `agent_eval.py` | Chạy agent thật trên một split với một cấu hình có tên; lưu trace, metric tự động, khung nhãn |
| `review_view.py` | In câu trả lời cạnh gold và chunk được trích để chấm tay (không tự gán nhãn) |

## Chạy lại

```bash
# 1) Dịch vụ: pgvector Day 6 + Ollama
docker compose -p day6_student_starter -f <day6>/starter/docker-compose.yml up -d --wait
docker compose -f starter/docker-compose.yml up -d ollama
docker exec day07-ollama ollama pull qwen3:4b-instruct

# 2) Python: dùng lại venv Day 6 (sentence-transformers, torch CPU, psycopg)
PY=C:/Users/<user>/.venvs/day06/Scripts/python.exe
cp .env.example .env

# 3) Dữ liệu và benchmark
$PY -m src.prepare_data

# 4) Retrieval trên dev (câu gốc), có và không có filter người dùng cung cấp
$PY -m src.retrieval_eval --split dev
$PY -m src.retrieval_eval --split dev --no-filter

# 5) Agent trên dev cho từng cấu hình (xem CONFIGS trong agent_eval.py)
$PY -m src.agent_eval --split dev --config A0_dense
$PY -m src.review_view --split dev --config A0_dense     # chấm tay -> reviews.json
$PY starter/agent_metrics.py --cases data/benchmark/cases.json \
    --traces results/agent/dev/A0_dense/traces.json \
    --reviews results/agent/dev/A0_dense/reviews.json --split dev \
    --output results/agent/dev/A0_dense/summary.json

# 6) Test: chỉ chạy được khi đã có results/selection.json
PYTHONUTF8=1 $PY -m unittest discover -s tests -v
```

Chi phí dịch vụ = 0 (embedding và LLM chạy local). Thời gian và token lấy từ trace (`elapsed_s`, `prompt_tokens` của Ollama).
