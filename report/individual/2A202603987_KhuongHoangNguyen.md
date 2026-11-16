# Báo cáo Day 7 — Khuong Hoang Nguyen (2A202603987)

> Trạng thái: hoàn thành tới mốc **2:00–3:30** (thử nghiệm và chọn cấu hình trên dev). Cấu hình đã chốt trong [results/selection.json](../../results/selection.json) **trước mọi lần chạy test Day 7**. Các ô "test" để trống cho mốc 3:30–4:00.
>
> Nhãn thủ công (answer grade, citation support, abstention) là **bản nháp do Claude (AI) chấm** theo thang BENCHMARK.md, có lý do từng câu trong `results/agent/dev/*/reviews.json`; học viên cần đọc lại và xác nhận.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** dùng lại corpus Day 6 — `docs/222_QLSXKS_L06_Nhom_2.pdf`, báo cáo BTL môn Quản lý sản xuất của nhóm học viên về Công ty CP Nước khoáng Quảng Ninh (quyền sử dụng như manifest Day 6). 26 trang nội dung, gồm text PDF và 14 ảnh bảng/sơ đồ đã OCR. Index **B** của Day 6 (chia theo heading/đoạn, ≤800 ký tự): **91 chunk / 26 trang**, xuất từ pgvector ra [data/day6/chunks.json](../../data/day6/chunks.json) bằng `src/prepare_data.py` (ID chunk giữ nguyên như Day 6).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8.7 (container Day 6, bảng `rag_chunks`, `strategy='B'`, cosine `<=>`, exact search). Query embedding: `google/embeddinggemma-2` rev `f9972ac`, 768 chiều, float32, chuẩn hoá L2, prompt query `task: search result | query: ` — trùng với cấu hình index Day 6; `src/retrievers.py` kiểm tra `embedding_model` của index trước khi truy vấn. Kiểm chứng: dense retrieval trên dev cho 15/17 Hit@1, 17/17 Hit@3, khớp đúng kết quả cấu hình C của Day 6.
- **Agent model, cách chạy và phân chia dev/test:** `qwen3:4b-instruct` (Q4_K_M, 4B) qua Ollama trong Docker, tool calling gốc, `temperature=0`, `seed=7`, `num_ctx=6144`; chạy local trên RTX 3050 Ti 4 GB (~77% GPU), chi phí $0. Bản `qwen3:4b` (thinking) đã thử và bỏ vì vẫn sinh suy luận dài dù tắt thinking (~1 phút/lượt gọi). Split giữ nguyên từ benchmark Day 6 (đã chốt ngày 2026-11-14): **dev 17 câu có đáp án + 2 câu thiếu evidence; test 7 + 2** (28 câu). Kết quả test của Day 6 đã được xem ở Day 6 nhưng không dùng để chỉnh bất kỳ phần nào của Day 7.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án: `gold = [{source_id, page, evidence}]`, hit khi chunk đúng nguồn + trang và chứa nguyên evidence (bỏ qua khác biệt khoảng trắng/hoa thường — 20/24 khớp nguyên văn, 4 câu chỉ khớp sau chuẩn hoá khoảng trắng). Hai câu test (q20, q21) có evidence vắt qua 2 chunk liên tiếp của index B nên chuyển thành câu nhiều evidence (`gold_mode: "all"`, phải có đủ cả hai chunk). Đây là sửa nhãn cho đúng định dạng dữ liệu, phát hiện lúc kiểm tra định dạng, không đụng vào cấu hình. 4 câu thiếu evidence (u01 số lao động, u02 giá bán lẻ — dev; u03 tổng giám đốc, u04 vốn điều lệ — test) đều sát chủ đề; đã xác nhận vắng mặt bằng regex trên **toàn bộ chunk của cả index A và B, gồm OCR** (`absence_check` trong [cases.json](../../data/benchmark/cases.json), kết quả trong [verification.json](../../data/benchmark/verification.json)), không dựa vào việc top-k không tìm thấy.

## Retrieval và thử nghiệm

Ba phương pháp retrieval, cùng corpus, index, câu hỏi, nhãn và cùng filter `section` do người dùng cung cấp (3 câu dev):

1. **Dense**: pgvector Day 6.
2. **Lexical BM25** (k1 = 1,5, b = 0,75): tách theo âm tiết tiếng Việt; biến thể `bm25_bigram` thêm bigram âm tiết để gần với từ ghép.
3. **Hybrid**: Reciprocal Rank Fusion (k = 60) của dense top-20 và BM25 top-20. Hợp nhất theo hạng nên không cần chuẩn hoá điểm cosine với điểm BM25.

**Retrieval trên câu hỏi gốc (không qua agent), dev, top-3, budget 2600 ký tự** — [results/retrieval/dev/](../../results/retrieval/dev/):

| Phương pháp | Filter | Hit@1 | Evidence Hit@3 | MRR@3 | Context Evidence | Median ms/query |
|---|---|---|---|---|---|---|
| dense | có | 15/17 (88%) | 17/17 (100%) | 0,941 | 17/17 | 133 |
| bm25 | có | 14/17 (82%) | 16/17 (94%) | 0,873 | 16/17 | 0,4 |
| bm25_bigram | có | 14/17 (82%) | 16/17 (94%) | 0,882 | 16/17 | 0,5 |
| hybrid (dense + bm25_bigram) | có | 15/17 (88%) | 17/17 (100%) | 0,941 | 17/17 | 151 |
| dense | không | 14/17 (82%) | 17/17 (100%) | 0,902 | 17/17 | 139 |
| hybrid (dense + bm25_bigram) | không | 15/17 (88%) | 17/17 (100%) | 0,941 | 17/17 | 145 |

Trên câu gốc, dev gần chạm trần: hybrid ngang dense khi có filter. Hybrid tốt hơn khi không có filter (q09 vẫn ở top-1). BM25 thua ở câu diễn đạt lại (q12: không có từ "xuất khẩu"). Thời gian dense/hybrid gồm cả embedding query trên CPU (~130 ms).

**Agent trên dev** (cùng vòng lặp, cùng prompt; Hit@1/@3 lấy từ lượt tìm đầu tiên của agent; context là context cuối cùng gửi cho model) — [results/dev_comparison.md](../../results/dev_comparison.md):

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `A0_dense` | Dense, k=3, filter người dùng, không rerank, 2600 ký tự, `append` | Agent + dense Day 6 | dev | 13/17 (76%) | 14/17 (82%) | 14/17 (82%) | median 15,3 s/câu; 1903 prompt tok; $0 |
| Phương pháp 2 `A1_bm25` | BM25 bigram, còn lại như A0 | Lexical bắt số liệu, tên riêng | dev | 12/17 (71%) | 15/17 (88%) | 15/17 (88%) | 13,8 s; 1893 tok; $0 |
| Phương pháp 3 `A2_hybrid` | RRF(dense, BM25 bigram), còn lại như A0 | Hai retriever bù lỗi cho nhau khi query bị viết lại | dev | 14/17 (82%) | 16/17 (94%) | 16/17 (94%) | 15,0 s; 1901 tok; $0 |
| Cải thiện 1 `A3_hybrid_orig` | A2 + lượt tìm đầu luôn dùng nguyên văn câu hỏi người dùng | Lỗi 1: truy vấn agent viết lại làm mất evidence | dev | 15/17 (88%) | 17/17 (100%) | 17/17 (100%) | **8,3 s; 1228 tok**; $0 |
| Cải thiện 2 `A4_hybrid_orig_ctx` | A3 + chunk kế tiếp, budget 4500, `best_rank`, prompt yêu cầu tìm lại | Lỗi 2: danh sách bị cắt sang chunk kế tiếp | dev | 15/17 (88%) | 17/17 (100%) | 17/17 (100%) | 10,6 s; 1806 tok; $0 |
| Cấu hình đã chọn `A3_hybrid_orig` | như trên | | test | *(mốc 3:30–4:00)* | | | |
| Baseline `A0_dense` | như trên | | test | *(mốc 3:30–4:00)* | | | |

A2 vừa là phương pháp mới vừa là cải thiện so với A0. A3 và A4 là hai cải thiện có giả thuyết riêng, xây trên A2.

**Lỗi dev 1 — truy vấn agent viết lại làm mất evidence (q02, q15, q19).** Ở A0, agent tự viết truy vấn, thường thêm tên công ty, ví dụ q19: *"các rủi ro được xếp vào mức nghiêm trọng … của công ty Cổ phần Nước khoáng Quảng Ninh"*. Tên công ty xuất hiện ở khắp tài liệu nên làm loãng embedding: gold p028-000 rơi khỏi top-3, trong khi câu gốc đưa nó lên **hạng 1** (cả 3 câu đều như vậy). Hậu quả: q02 bị từ chối nhầm; q15 và q19 bị trả lời **bịa** bằng cách ghép các đoạn không liên quan, citation không hỗ trợ. Thử 1 (A2, hybrid): BM25 kéo lại q02 và q15, q19 vẫn sai. Thử 2 (A3, lượt đầu dùng câu gốc): **17/17 câu điểm 2**, Context Evidence 17/17.

**Lỗi dev 2 — đáp án dạng danh sách bị cắt sang chunk kế tiếp (q09).** Chunk gold p013-000 có 4/5 nguyên tắc; nguyên tắc thứ 5 ("đảm bảo tính linh hoạt") nằm ở p013-001. Theo metric, context vẫn "có evidence", nhưng câu trả lời chỉ đúng một phần (điểm 1 ở A0–A2). Thử (A4): thêm chunk kế tiếp của mỗi chunk được chọn. Kết quả: q09 đủ 5 ý. Tuy nhiên A3 cũng trả lời đủ q09, vì câu gốc tình cờ đưa p013-001 vào top-3. Trên dev, A4 vì vậy không hơn A3 về chất lượng mà tốn thêm 47% prompt token và 27% thời gian. Đây là kết quả "không cải thiện" trên dev và được ghi nhận như vậy.

**Lỗi phụ đã ghi nhận:** A1 q19 đọc bảng OCR p027-003 (chỉ số IxPxD) rồi gán nhãn "nghiêm trọng" cho các con số mà bảng không ghi như vậy. A1 q24 bỏ ý "lọc thô" dù ý này có trong context.

**Tiêu chí chọn và lý do** (ghi trong `src/select_config.py` và `results/selection.json`, áp dụng trên dev trước khi chạy test): ① điểm grounded trung bình → ② từ chối nhầm/từ chối đúng → ③ tỷ lệ citation hợp lệ → ④ Context Evidence Rate → ⑤ median thời gian, rồi prompt token. A3 và A4 hoà ở ①–④ (2,00/2; 0 từ chối nhầm; 18/18 citation; 17/17). A3 thắng ở ⑤ (8,3 s và 1228 token so với 10,6 s và 1806 token) nên **chọn A3**. Vì nhiều nhãn ①–④ đã bão hoà trên dev, lựa chọn thực chất do chi phí quyết định; rủi ro là A3 không có cơ chế chunk lân cận cho danh sách dài.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool:** một tool `search_documents(query)`. Filter `section` lấy từ phạm vi người dùng chọn, model không tự đặt filter.
  - **Giới hạn và điều kiện dừng:** tối đa 3 lượt tìm và 6 lượt gọi LLM mỗi câu. Hết lượt tìm thì không cung cấp tool nữa và yêu cầu model trả lời hoặc từ chối. Dừng khi model trả lời, khi model trả lời `KHÔNG ĐỦ BẰNG CHỨNG`, hoặc khi chạm giới hạn lượt gọi.
  - **Guard:** nếu model trả lời trước khi tìm, câu trả lời bị từ chối và model được nhắc gọi tool, tối đa 1 lần. Ở A0–A2 guard này kích hoạt **9/19 câu**: model định trả lời bằng kiến thức riêng.
  - **Context:** dedupe theo ID, gộp kết quả của mọi lượt tìm, cắt theo budget ký tự (kèm header `[id | nguồn | trang | chương]`). Chính sách `append` giữ thứ tự xuất hiện; `best_rank` ưu tiên hạng tốt nhất. Model chỉ thấy context tích luỹ mới nhất, không nhân budget.
  - **Citation:** chỉ chấp nhận `[chunk-id]` có trong context cuối. Không còn citation hợp lệ thì chuyển sang từ chối. ID hợp lệ vẫn phải được đọc đối chiếu, xem bảng dưới.
  - **Lỗi Ollama/pgvector:** ghi `status: "error"` kèm traceback, không thay bằng kết quả giả. Dev không có lỗi nào.
  - **Ở A3/A4 model không tự gọi thêm tool lần nào:** 19/19 câu kết thúc sau 1 lượt gọi LLM, sau lượt tìm câu gốc do hệ thống thực hiện. Vai trò của model ở đây là quyết định bằng chứng đã đủ hay chưa (có quyền tìm thêm) và trả lời. Tool use 19/19 ở bảng dưới tính cả lượt tìm hệ thống này.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A0_dense / dev | 19/19 | 1 lượt tìm/câu; 2,47 LLM call | 3/1/13 — 1,59 | 15/19 | 16/16 | 2/2 | 1/17 |
| A1_bm25 / dev | 19/19 | 1; 2,47 | 2/2/13 — 1,65 | 15/16 | 16/16 | 2/2 | 1/17 |
| A2_hybrid / dev | 19/19 | 1; 2,47 | 1/1/15 — 1,82 | 16/17 | 17/17 | 2/2 | 0/17 |
| **A3_hybrid_orig / dev** | 19/19 | 1 (hệ thống); 1,00 | 0/0/17 — 2,00 | 18/18 | 17/17 | 2/2 | 0/17 |
| A4_hybrid_orig_ctx / dev | 19/19 | 1 (hệ thống); 1,00 | 0/0/17 — 2,00 | 18/18 | 17/17 | 2/2 | 0/17 |
| A3 / test, A0 / test | *(mốc 3:30–4:00)* | | | | | | |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q19 (A3, dev) | trực tiếp | `search("Trong đánh giá rủi ro, những rủi ro nào được xếp vào mức nghiêm trọng?")` → p027-003 (bảng OCR), **p028-000**, p027-002 | "rủi ro trong khâu sản xuất, quá trình cạnh tranh, vận hành sản xuất, pháp lý [p028-000]" | 2 — khớp nguyên văn gold; citation hỗ trợ |
| u01 (A3, dev) | thiếu evidence | `search(câu gốc)` → p004-001, p007-001, p015-003 (không có số lao động) | "KHÔNG ĐỦ BẰNG CHỨNG: Không có thông tin về số lượng lao động…" | từ chối đúng; không bịa số |
| q19 (A0, dev) — **lỗi** | trực tiếp | `search("… của công ty Cổ phần Nước khoáng Quảng Ninh")` → p026-003, p024-002, p023-000 (không có gold) | liệt kê "rủi ro sản phẩm không đạt chất lượng…" [p024-002][p026-003] | 0 — sai; 2 citation đều không hỗ trợ việc "xếp vào mức nghiêm trọng" |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace từng câu (tool call, query, filter, kết quả kèm score/rank, context cuối, câu trả lời, citation, token, thời gian): `results/agent/dev/<cfg>/traces.json`.
  - Metric tự động: `auto.json`. Nhãn kèm lý do: `reviews.json`. Tổng hợp bằng `starter/agent_metrics.py`: `summary.json`.
  - Retrieval: `results/retrieval/dev/<method>/traces.json`.
  - Lệnh chạy: [src/README.md](../../src/README.md).
  - Tổng thời gian chạy agent trên 19 câu dev: A0 346 s, A3 205 s. Chi phí dịch vụ $0.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau:** baseline A0 đạt 1,59/2 (14/17 Context Evidence). A3 đạt 2,00/2 (17/17) và nhanh hơn 46%: một lượt gọi LLM thay vì 2,5, vì không còn vòng "trả lời → bị từ chối → tìm".
  - **Giới hạn 1 — dev chạm trần:** dev gần trần nên không phân biệt được A3 với A4.
  - **Giới hạn 2 — mỗi cấu hình chỉ chạy 1 lần:** temperature 0, seed cố định, chưa đo độ dao động.
  - **Giới hạn 3 — corpus nhỏ:** chỉ một tài liệu nên filter chỉ có `section`.
  - **Giới hạn 4 — nhãn do AI chấm:** chưa có người thứ hai.
  - **Giới hạn 5 — agent hầu như không tự tìm lại:** chưa có lần nào model chủ động tìm lượt 2, kể cả với prompt A4. Agent đang phụ thuộc vào chất lượng lượt tìm đầu tiên.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
