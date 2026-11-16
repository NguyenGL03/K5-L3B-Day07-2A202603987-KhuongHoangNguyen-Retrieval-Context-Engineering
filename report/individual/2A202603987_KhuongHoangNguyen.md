# Báo cáo Day 7 — Khuong Hoang Nguyen (2A202603987)

> Trạng thái: hoàn thành. Cấu hình đã chốt trong [results/selection.json](../../results/selection.json) (2026-11-16 00:57) **trước mọi lần chạy test Day 7**. Test chạy đúng 1 lần cho 2 cấu hình (baseline A0 và cấu hình đã chọn A3), không chỉnh gì sau khi xem kết quả test.
>
> Nhãn thủ công (answer grade, citation support, abstention) là bản nháp do Claude (AI) chấm theo thang BENCHMARK.md, có lý do từng câu trong `results/agent/{dev,test}/*/reviews.json`, đã được tôi đọc lại và xác nhận.

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
| Cấu hình đã chọn `A3_hybrid_orig` | như trên | | test | 4/7 (57%) | 5/7 (71%) | 5/7 (71%) | median 6,2 s/câu; 1261 prompt tok; $0 |
| Baseline `A0_dense` | như trên | | test | 4/7 (57%) | 5/7 (71%) | 5/7 (71%) | median 16,0 s/câu; 1839 prompt tok; $0 |

**Retrieval trên câu hỏi gốc, test** (chạy sau khi chốt cấu hình, chỉ để giải thích kết quả, không dùng để chọn) — [results/retrieval/test/](../../results/retrieval/test/): dense 4/7 Hit@1, 6/7 Hit@3, Context Evidence 6/7; bm25_bigram 4/7, 5/7, 5/7; hybrid 4/7, 5/7, 5/7. Trên test, hybrid **kém dense 1 câu** (q20, câu hỏi rất ngắn "Có những nguyên tắc nào cần tuân theo?": dense cho p022-001, p023-003, p022-002 — đủ 2 gold; BM25 xếp hạng 1 đoạn dẫn p022-000 "Các nguyên tắc … bao gồm:" và hạng 2 p023-002, nên RRF cho p022-001, p022-000, p023-002 và đẩy gold p022-002 ra khỏi top-3). Lợi thế của hybrid trên dev không lặp lại trên test.

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
| A0_dense / test | 9/9 | 1 lượt tìm/câu; 2,44 LLM call (guard kích hoạt 4/9) | 1/2/4 — 1,43 | 5/10 | 7/7 | 2/2 | 0/7 |
| **A3_hybrid_orig / test** | 9/9 | 1 (hệ thống); 1,00 | 0/3/4 — 1,57 | 6/9 | 7/7 | 2/2 | 0/7 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q19 (A3, dev) | trực tiếp | `search("Trong đánh giá rủi ro, những rủi ro nào được xếp vào mức nghiêm trọng?")` → p027-003 (bảng OCR), **p028-000**, p027-002 | "rủi ro trong khâu sản xuất, quá trình cạnh tranh, vận hành sản xuất, pháp lý [p028-000]" | 2 — khớp nguyên văn gold; citation hỗ trợ |
| u01 (A3, dev) | thiếu evidence | `search(câu gốc)` → p004-001, p007-001, p015-003 (không có số lao động) | "KHÔNG ĐỦ BẰNG CHỨNG: Không có thông tin về số lượng lao động…" | từ chối đúng; không bịa số |
| q19 (A0, dev) — **lỗi** | trực tiếp | `search("… của công ty Cổ phần Nước khoáng Quảng Ninh")` → p026-003, p024-002, p023-000 (không có gold) | liệt kê "rủi ro sản phẩm không đạt chất lượng…" [p024-002][p026-003] | 0 — sai; 2 citation đều không hỗ trợ việc "xếp vào mức nghiêm trọng" |
| q21 (A3, test) — **lỗi** | metadata filter, 2 evidence | `search(câu gốc)` + filter chương chuỗi cung ứng → **p014-001**, p018-002, p014-000; thiếu p014-002 | 3 ưu điểm của hợp đồng sản xuất [p014-001] | 1 — đúng 3/4 ý; ý "tổng chi phí thấp hơn" nằm ở chunk kế tiếp p014-002 (đúng lỗi dev 2) |
| q21 (A0, test) — **lỗi** | metadata filter, 2 evidence | `search("ưu điểm chính của quản lý chuỗi cung ứng tại Công ty …")` → p018-000, p016-000, p015-003 (không có gold) | "ưu điểm" tự tổng hợp từ đoạn phân phối, nhà cung cấp [3 citation] | 0 — sai; citation không nói về ưu điểm (đúng lỗi dev 1) |
| q04 (A0 và A3, test) | diễn đạt lại | context có cả p006-000 (gold, "161%") và p006-001 (bảng OCR kế hoạch) | "161% [p006-001]" | 1 — số đúng nhưng trích nhầm bảng OCR không chứa 161% |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace từng câu (tool call, query, filter, kết quả kèm score/rank, context cuối, câu trả lời, citation, token, thời gian): `results/agent/dev/<cfg>/traces.json`.
  - Metric tự động: `auto.json`. Nhãn kèm lý do: `reviews.json`. Tổng hợp bằng `starter/agent_metrics.py`: `summary.json`.
  - Retrieval: `results/retrieval/dev/<method>/traces.json`.
  - Lệnh chạy: [src/README.md](../../src/README.md).
  - Tổng thời gian chạy agent trên 19 câu dev: A0 346 s, A3 205 s; trên 9 câu test: A0 164 s, A3 92 s. Chi phí dịch vụ $0.
  - Test: `results/agent/test/{A0_dense,A3_hybrid_orig}/` (cùng cấu trúc file như dev).
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau:** baseline A0 đạt 1,59/2 (14/17 Context Evidence). A3 đạt 2,00/2 (17/17) và nhanh hơn 46%: một lượt gọi LLM thay vì 2,5, vì không còn vòng "trả lời → bị từ chối → tìm".
  - **Test (A0 → A3):** grounded 1,43 → 1,57/2; citation hợp lệ 5/10 → 6/9; từ chối đúng 2/2 ở cả hai, không từ chối nhầm; median 16,0 → 6,2 s/câu; prompt token −31%. Context Evidence bằng nhau (5/7): cả hai đều trượt q20 và q21, hai câu cần đủ 2 chunk liên tiếp. A3 cải thiện q21 từ 0 lên 1 (lượt tìm câu gốc tìm được đúng phần, A0 viết lại query và trả lời bịa — lỗi dev 1 lặp lại trên test).
  - **Khoảng cách dev–test:** A3 giảm từ 2,00 (dev) xuống 1,57 (test). Mất điểm chủ yếu ở danh sách vắt qua 2 chunk (q20, q21) — đúng lỗi dev 2 mà A4 (thêm chunk kế tiếp) nhắm tới. Trên dev, A4 không hơn A3 nên bị loại vì tốn hơn; test cho thấy dev quá ít câu dạng này để đánh giá cải thiện đó. Không chạy A4 trên test để giữ nguyên tắc chỉ chạy cấu hình đã chốt; đây là điều nên thêm vào dev ở vòng sau (thêm câu danh sách dài vào dev, không lấy từ test).
  - **Lỗi citation chung (q04):** cả hai cấu hình trích bảng OCR thay vì câu văn chứa số liệu. Cơ chế kiểm tra citation chỉ xác nhận ID có trong context, không kiểm tra nội dung, nên lỗi này không bị chặn.
  - **Giới hạn 0 — test nhỏ:** 7 câu có đáp án; chênh lệch 1 câu = 0,14 điểm, chưa đủ để khẳng định A3 tốt hơn A0 về chất lượng; lợi thế rõ ràng là thời gian và token.
  - **Giới hạn 1 — dev chạm trần:** dev gần trần nên không phân biệt được A3 với A4.
  - **Giới hạn 2 — mỗi cấu hình chỉ chạy 1 lần:** temperature 0, seed cố định, chưa đo độ dao động.
  - **Giới hạn 3 — corpus nhỏ:** chỉ một tài liệu nên filter chỉ có `section`.
  - **Giới hạn 4 — nhãn do AI chấm:** chưa có người thứ hai.
  - **Giới hạn 5 — agent hầu như không tự tìm lại:** chưa có lần nào model chủ động tìm lượt 2, kể cả với prompt A4. Agent đang phụ thuộc vào chất lượng lượt tìm đầu tiên.

## Thử thách nâng cao (nếu thực hiện)

> **Kết quả: test vòng 2 (bộ test mới) đạt 27/30 = 90,0%, đúng ngưỡng 90% (cần ít nhất 27/30).** Bộ test vòng 1 (20/30) đã bị nhìn thấy nên chuyển thành dữ liệu tinh chỉnh; theo đề, điểm báo cáo là của bộ test mới, được viết sau khi chốt cấu hình, khoá bằng sha256 và chạy đúng 1 lần.
> Nhãn đạt/không đạt là bản nháp do Claude (AI) chấm, đối chiếu với PDF gốc. **Học viên cần đọc lại từng nhãn**, nhất là m2_03 (xem phần lỗi), vì kết quả nằm đúng ở ngưỡng: chỉ cần một nhãn đổi từ đạt sang không đạt là xuống 26/30.

### Tài liệu và index

- **Tài liệu:** *Public Bank Berhad 2025 Integrated Annual Report, Financial Statements*.
  - 328 trang, PDF số do InDesign tạo ngày 2026-04-01. Số trang PDF = số in + 2.
  - Báo cáo công khai; tải bản gốc từ trang Investor Relations của Public Bank. **Tệp PDF không được commit.**
- **Pipeline:** PyMuPDF đọc thẳng lớp text và hình vector, không cần OCR.
  - Bảng: ghép dòng theo toạ độ, mang theo header cột, xoay lại bảng nằm ngang.
  - 12 biểu đồ cột: ghép giá trị với năm theo toạ độ x.
  - 10 donut: ghép nhãn với lát theo thứ tự vòng và nối màu lát với chú giải.
  - 7 sơ đồ (6 chu trình rủi ro/ICAAP, 1 sơ đồ luồng): ghép chữ cong, gán bullet vào giai đoạn gần nhất.
  - Mỗi hình có ảnh crop và bbox.
- **Vòng 2: trường `display`** cho agent đọc. Text được index (và embedding) giữ nguyên từng byte, kiểm tra lại sau mỗi lần build, nên không phải embed lại.
  - Bảng rộng (≥ 5 cột số): mỗi giá trị kèm nhãn cột ghép từ header nhiều tầng, ví dụ `Hong Kong SAR = 1,182,608`. Kiểm tra: 767 dòng có cột Total cộng đúng bằng tổng; mọi dòng lệch đều do có cột subtotal (Note 54, hợp đồng bảo hiểm).
  - Chú thích dưới biểu đồ (`*`, `#`) gắn vào chunk biểu đồ.
  - Vị trí các khối trong sơ đồ chu trình (trên/dưới/trái/phải).
  - Chunk bảng tiếp nối mang theo dòng đơn vị của trang, ví dụ `(RM’MILLION)`.
- **Kiểm chứng trích xuất với bản gốc:**
  - Donut tr. 11: 65/65 giá trị khớp bảng tr. 10.
  - Biểu đồ cột tr. 8–9: khớp bảng 5 năm tr. 6–7 và ảnh render.
  - Donut tr. 5 và 7 sơ đồ: đối chiếu từng giá trị với ảnh render.
  - Bảng xoay tr. 247: tổng các cột khớp ô tổng.
- **Phần không index:** tr. 327 trống; 4 ảnh raster chỉ để trang trí (bìa, mockup, ảnh stock).
- **Hai index trên pgvector Day 6** (`embeddinggemma-2`, 768 chiều): `PBB` (chunk 1.200 ký tự, 939 chunk) và `PBB600` (600 ký tự, 1.631 chunk). Cấu hình cuối dùng `PBB600`, hybrid, k=6.

### Benchmark

| Bộ | Số câu | Vai trò | Khoá |
|---|---|---|---|
| Dev | 19 (15 có đáp án + 4 thiếu bằng chứng) | chọn cấu hình vòng 1 | [FROZEN.json](../../advanced/benchmark/FROZEN.json), 11:53 |
| Test vòng 1 | 30 (6 bảng, 5 sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text, 6 thiếu bằng chứng) | chạy 1 lần (20/30); sau đó thành dữ liệu tinh chỉnh | cùng file trên |
| `tune` = dev + test vòng 1 | 49 | chọn cấu hình vòng 2 | — |
| **Test vòng 2 (`test2`)** | **30**, cùng cơ cấu | **điểm báo cáo**, chạy 1 lần | [FROZEN_test2.json](../../advanced/benchmark/FROZEN_test2.json), sha256 `3f22064d…`, 23:07:39 |

- **Kiểm chứng tự động** ([verify_benchmark.py](../../advanced/verify_benchmark.py), `--test2`):
  - mọi evidence text/bảng có mặt trên đúng trang của PDF gốc;
  - mọi regex vắng mặt có 0 kết quả trên cả 328 trang;
  - không câu test2 nào trùng câu hỏi hoặc trùng dữ kiện đáp án với vòng 1.
- **Cách viết test2 để tránh chọn câu dễ** ([sample_test2.py](../../advanced/benchmark/sample_test2.py), [build_test2.py](../../advanced/benchmark/build_test2.py)):
  - Mục tiêu được rút ngẫu nhiên (seed 20261116) từ các trang/hình chưa dùng ở vòng 1: 6 bảng (mỗi note một bảng), 5 biểu đồ, 5 sơ đồ, 4 dòng neo cho multi-hop, 4 trang văn xuôi.
  - Phần tử hỏi trong mỗi sơ đồ cũng được rút theo seed (20261117).
  - Bảng rút trúng gồm cả bảng rộng 10 cột và dòng tổng không nhãn (t2_06). Câu hỏi được viết sau khi đã có `selection_r2.json`.
  - Có 2 bước loại câu: ATM (p20 có "2,000 self-service terminals") và tổng nợ phải trả của Group, vì p6 chứa cả hai số nên không còn là multi-hop.

### Vòng 1: chọn trên dev, test 20/30

- Agent `qwen3:4b-instruct` (local, $0), 12 cấu hình trên dev; chọn H (17/19) theo [selection.json](../../advanced/results/selection.json).
- Test vòng 1, chạy 1 lần: **20/30 (66,7%)**.

| Nhóm | Bảng | Sơ đồ | Biểu đồ | Multi-hop | Text | Thiếu bằng chứng |
|---|---|---|---|---|---|---|
| Đạt | 3/6 | 4/5 | 3/5 | 0/4 | 4/4 | 6/6 |

- Nguyên nhân lỗi:
  - mất đơn vị nghìn (RM'000);
  - truy vấn kế hoạch gắn nhầm tên báo cáo cho multi-hop;
  - sơ đồ không có vị trí;
  - chú thích `#` không gắn vào biểu đồ;
  - model 4B không gióng được cột bảng xoay 15 cột.
- Trace và nhãn: [results/agent/test/](../../advanced/results/agent/test/H_hybrid600_v2_pagehead_operands/).

### Vòng 2: tinh chỉnh trên `tune` (49 câu) và chọn cấu hình

Mỗi thay đổi nhắm vào một loại lỗi đã thấy. Mọi cấu hình đều chạy thật trên cả 49 câu và được chấm tay ([results/agent/tune/](../../advanced/results/agent/tune/)).

| Cấu hình | Thay đổi | Đạt /49 | Multi-hop | Từ chối nhầm | Median s/câu | Chi phí |
|---|---|---|---|---|---|---|
| H (vòng 1) | — | 37 | 1/7 | — | — | $0 |
| J | `display`, `unit_fix` (thêm "thousand" cho số lấy từ bảng RM'000), chỉ coi là từ chối khi câu trả lời *bắt đầu* bằng INSUFFICIENT EVIDENCE | 41 | 2/7 | 5 | 17,2 | $0 |
| K | J + kế hoạch truy vấn không tự gắn tên báo cáo, tìm lại trước khi từ chối, 6 lượt tìm | 40 | 2/7 | 3 | 20,1 | $0 |
| M / L | J / K + mang dòng đơn vị sang chunk tiếp nối, quy tắc "lấy đúng nguồn được nêu", xem lại khi từ chối sau `calculate` | 41 / 41 | 2/7 / 3/7 | 4 / 2 | 17,7 / 25,1 | $0 |
| N / P | M / L + `unit_fix` xét từng số | 40 / 42 | 2/7 / 3/7 | 4 / 2 | 14,8 / 15,9 | $0 |
| **S (chọn)** | P với **`gemini-3.5-flash-lite`** (API) + guard số chấp nhận số đổi đơn vị (×/÷1.000), không chạy lại `calculate` trùng, buộc trả lời khi hết lượt tìm | **49** | **7/7** | **0** | **7,8** | $0,19* |

\* Giá paid tier tra ngày 16/11/2026: $0,30 / 1M token vào, $2,50 / 1M token ra (tính cả thinking). Nếu key dùng free tier thì chi phí thực là $0.

- **Đổi model:** qwen3:4b dừng ở khoảng 84–86% trên `tune`; các câu còn sai là lỗi suy luận (lấy nhầm số, không tính, từ chối dù có dữ kiện).
  - Bản qwen3:4b có thinking bị timeout (>300 s/lời gọi), đã bỏ.
  - Học viên chọn Gemini 3.5 Flash Lite trước khi viết test2. Client: [llm_gemini.py](../../advanced/llm_gemini.py).
- **Tiêu chí chọn** ([selection_r2.json](../../advanced/results/selection_r2.json)), ghi trước khi viết test2: tỷ lệ đạt → từ chối nhầm → lỗi hệ thống → median latency → token.
- **Lưu ý:** 49/49 là lạc quan, vì 49 câu này chính là dữ liệu đã dùng để tìm và sửa lỗi.
- **Quy tắc đơn vị vòng 2:** giá trị từ bảng RM'000 đạt khi có "thousand", "(RM'000)" hoặc được đổi đúng (vd. RM9,583,686,000). Theo quy tắc này, t04 của test vòng 1 sẽ đạt, tức vòng 1 sẽ là 21/30. Nhãn vòng 1 được giữ nguyên.

### Kết quả test vòng 2 (S, chạy 1 lần)

| Nhóm | Đạt |
|---|---|
| Bảng | 4/6 |
| Ảnh, sơ đồ | 5/5 |
| Biểu đồ | 5/5 |
| Multi-hop | 4/4 |
| Text | 3/4 |
| Thiếu bằng chứng | 6/6 |
| **Tổng** | **27/30 (90,0%)** |

- Lỗi hệ thống / timeout: 0. Từ chối nhầm: 1/24 (t2_06).
- Median 5,5 s/câu, tổng 450 s; trung bình 4,07 lời gọi LLM và 13.987 prompt token mỗi câu.
- Chi phí $0,14 cho 30 câu theo giá paid tier ($0 nếu free tier).
- Retrieval trên câu gốc (hybrid_600, k=6): Hit@1 21/24, Hit@6 24/24, đủ trang bắt buộc 24/24.
- Trace: [results/agent/test2/S_R_scaled_dedupe/](../../advanced/results/agent/test2/S_R_scaled_dedupe/). Gồm truy vấn, kết quả truy hồi, context cuối, `calculate`, các bước guard, citation, thời gian, token và chi phí cho từng câu; nhãn và lý do trong `reviews.json`.

### Phân tích lỗi test vòng 2

1. **t2_02, lỗi do bước sửa đơn vị của pipeline.**
   - Bản nháp của model đúng: "RM1,642,000", tức 1,642 nghìn đã đổi sang RM.
   - Bước `unit_fix` thêm "thousand" vì chuỗi 1,642,000 cũng xuất hiện trong một câu văn của chunk p69 có tiêu đề RM'000.
   - Kết quả thành "RM1,642,000 thousand", sai 1.000 lần.
   - Bài học: số trong câu văn (đã ghi bằng RM) không được coi như ô bảng RM'000.
2. **t2_06, lỗi truy hồi do cách chia chunk bảng rộng.**
   - Dòng tổng "Expenditure incurred" của bảng 10 cột p168 nằm ở chunk p168-t6.
   - Text được index của chunk này chỉ có số, không có nhãn dòng/cột, nên 6 lượt tìm đều không lấy được; model từ chối.
   - Trang 168 vẫn được tìm thấy (Hit theo trang), nhưng không phải đúng chunk.
3. **x2_04, giới hạn trích xuất bố cục nhiều cột.**
   - Phụ lục p274 có hai cột "công ty con | giám đốc" đặt cạnh nhau.
   - Ghép dòng theo toạ độ trộn hai cột, nên model nêu sai người.
4. **Lỗi đã biết nhưng không làm sai câu nào:** ghi chú bố cục ghi "Control & Mitigation (centre)", trong khi trên trang nó nằm ở dưới-trái. Lỗi này được phát hiện khi viết test2 và **không được sửa**, vì sửa lúc đó là chỉnh pipeline bằng test.
5. **Câu cần xem lại (m2_03):** giá trị và phép tính đúng, nhưng model gọi cột là "investment banking management", tức nhãn cột ghép sai từ bước trích xuất. Mình chấm đạt vì giá trị đúng cột Investment Banking.

### Giới hạn

- Kết quả nằm đúng ngưỡng; mỗi bộ test chỉ có 30 câu và mỗi cấu hình chạy 1 lần. Model API có thể cho kết quả khác giữa các lần chạy dù cố định seed, và chạy lại không phải là cách báo điểm hợp lệ.
- Gemini là dịch vụ bên ngoài: các đoạn trích của báo cáo công khai được gửi tới Google khi chạy. Key nằm trong `.env`, không commit.
- Không chỉnh gì sau khi xem test2. Lỗi `unit_fix` (t2_02), lỗi chia chunk (t2_06) và lỗi bố cục hai cột (x2_04) được ghi lại để làm việc tiếp, nhưng chưa sửa.
- Máy ngủ giữa chừng làm hỏng một lượt chạy ở vòng 1 (B: timeout, tính là lỗi).
