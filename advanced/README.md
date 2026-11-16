# Thử thách nâng cao: Public Bank Berhad 2025 Annual Report

## Tài liệu

| Mục | Giá trị |
|---|---|
| Tài liệu | *Public Bank Berhad: 2025 Integrated Annual Report, Financial Statements* (PDF do Adobe InDesign tạo, ngày 2026-04-01) |
| Tệp cục bộ | `docs/publicbank_fr-2025_hr.pdf` (6.942.868 byte, 328 trang). **Không commit tệp này**; người chấm tải bản gốc từ trang Investor Relations của Public Bank (mục Annual Reports, 2025). |
| Quyền | Báo cáo thường niên công khai của công ty niêm yết. Chỉ dùng để học tập; repo không phân phối lại tệp PDF hoặc ảnh crop. |
| Đánh số trang | `page` là số trang PDF (bắt đầu từ 1). Số in trên trang bằng `page - 2` (từ trang 3 trở đi). |

## Pipeline (`advanced/ingest.py`)

PDF có lớp text gốc, không phải ảnh scan, nên pipeline đọc thẳng text và hình vector bằng PyMuPDF thay vì OCR.

| Thành phần | Cách xử lý | Kiểm chứng |
|---|---|---|
| Text, bảng | Ghép các span thành dòng theo toạ độ; khoảng cách giữa các cột thành ` \| `. Các dòng header cột (Group/Bank, 2025/2024, RM'000) được lặp lại ở mỗi chunk của cùng trang. Bảng xoay 90° (ví dụ Note 54, tr. 245–248) được xoay về chiều đọc. | Bảng xoay tr. 247: tổng các mảng trong nước bằng 27.121.468 và tổng nước ngoài bằng 2.388.080, khớp đúng với ô tổng trong bảng. |
| Biểu đồ cột (12) | Lấy trục năm 2021–2025 và nhãn giá trị xoay dọc, ghép với nhau theo toạ độ x. Tiêu đề và đơn vị lấy từ chữ đậm phía trên. | Khớp ảnh render tr. 8, 9 và bảng 5 năm ở tr. 6–7. |
| Donut (10) | Ghép nhãn % với lát theo thứ tự vòng, bằng quy hoạch động trên các góc. Màu của lát được nối với màu ô chú giải. Lát nền (vẽ cả vòng tròn) và giá trị âm (không có lát) được xử lý riêng. | Tr. 11: 65/65 giá trị khớp bảng tr. 10. Tr. 5: khớp ảnh render; share capital 1,7% khớp phép tính từ bảng cân đối. |
| Sơ đồ chu trình (6) | Ghép lại các nhãn chữ cong (mỗi ký tự là một span). Gán danh sách bullet vào giai đoạn gần nhất; bullet là hình vẽ nên được dùng để xác định đầu mỗi mục. Các hộp giám sát in hoa được liệt kê riêng. | Khớp ảnh render tr. 278, 283, 318, 322, 323, 325. |
| Sơ đồ luồng (1) | Một hàng hộp tiêu đề, mỗi hộp có một cột chữ bên dưới; thứ tự đọc từ trái sang phải. | Khớp ảnh render tr. 278. |

Mỗi hình có một ảnh crop trong `data/crops/` và bbox trong metadata, đủ để mở lại đúng vùng trên bản gốc.

**Những phần không index:**
- Trang 327 trống, không có lớp text.
- 4 ảnh raster chỉ để trang trí: bìa tr. 1, mockup tr. 2, ảnh stock tr. 14.

**Giới hạn đã biết:**
- Một số bảng nhiều tầng header (ví dụ bảng xoay) bị gộp tên cột, như "Banking Management".
- Bảng nối sang trang sau chỉ mang header của trang hiện tại.

## Index

| | PBB (1.200 ký tự) | PBB600 (600 ký tự) |
|---|---|---|
| Số chunk | 939 (901 text/bảng, 10 bảng xoay, 12 cột, 10 donut, 7 sơ đồ) | 1.631 |
| Vector store | Bảng pgvector `rag_chunks` của Day 6, strategy `PBB`/`PBB600` | như cột bên |
| Embedding | `google/embeddinggemma-2` rev `f9972ac`, 768 chiều, prompt `document`/`query`, chuẩn hoá L2 | như cột bên |

## Benchmark (`benchmark/`)

- **Dev 19 câu** (15 có đáp án + 4 thiếu bằng chứng).
- **Test 30 câu**: 6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text, 6 thiếu bằng chứng.
- `verify_benchmark.py` kiểm tra hai việc:
  - mọi evidence dạng text/bảng có mặt trên đúng trang của **PDF gốc**;
  - mọi regex `absence_check` có 0 kết quả trên cả 328 trang.
- Benchmark được khoá trong `benchmark/FROZEN.json` (sha256, 2026-11-16 11:53) trước mọi lần chạy thử.

## Vòng 2: trường `display` (chỉ agent đọc)

`ingest.py` ghi thêm trường `display` cho mỗi chunk. Trường `text` được index giữ nguyên từng byte, nên embedding trong pgvector vẫn dùng được.

| Bổ sung | Cách làm | Kiểm chứng |
|---|---|---|
| Nhãn cột cho bảng rộng (≥ 5 cột số) | Gom cột theo mép phải các số; ghép các dòng header phía trên theo cột; header bắc qua nhiều cột (Group/Bank, 2025/2024) gán theo cột gần nhất khi nhãn bị trùng | 767 dòng có cột Total cộng đúng; dòng lệch đều do cột subtotal |
| Chú thích biểu đồ | Dòng `*`, `#`… dưới biểu đồ gắn vào biểu đồ có cùng ký hiệu | `#` tr. 9 gắn vào Total assets và Total capital ratio |
| Vị trí trong sơ đồ chu trình | Vị trí so với tâm vòng (trên/dưới/trái/phải) | Các khối giám sát đúng. **Lỗi đã biết:** giai đoạn "Control & Mitigation" bị ghi "centre" (thực tế dưới-trái); phát hiện khi viết test2 nên không sửa |
| Đơn vị cho chunk tiếp nối | Lặp lại dòng đơn vị gần nhất của trang, ví dụ `(RM’MILLION)` | 5 chunk tr. 4–10 |

**Giới hạn đã biết (từ test2):**
- Dòng tổng không nhãn của bảng rộng nằm riêng một chunk thì khó truy hồi (tr. 168).
- Phụ lục bố cục hai cột bị trộn cột (tr. 274).

## Vòng 2: model và cấu hình chọn

- Cấu hình `S_R_scaled_dedupe` = pipeline P (hybrid_600, k=6, kế hoạch truy vấn v3, display, các guard) với `gemini-3.5-flash-lite` qua Gemini API ([llm_gemini.py](llm_gemini.py)).
- Cần `GEMINI_API_KEY` trong `.env`. Key chỉ được gửi trong header và không bao giờ ghi vào log.
- Chi phí được ghi theo giá paid tier tra ngày 16/11/2026: $0,30 / 1M token vào, $2,50 / 1M token ra.
- Chọn trên `tune` (dev + test vòng 1, 49 câu), xem [results/selection_r2.json](results/selection_r2.json).
- Test vòng 2 ([benchmark/test2.json](benchmark/test2.json), 30 câu) được viết sau khi chốt cấu hình, với mục tiêu rút ngẫu nhiên theo seed ([benchmark/sample_test2.py](benchmark/sample_test2.py)). Bộ test được khoá ([benchmark/FROZEN_test2.json](benchmark/FROZEN_test2.json)) rồi chạy 1 lần: **27/30 = 90,0%**.

## Chạy lại

```bash
python -m advanced.ingest            # chunks.json, visuals.json, crops/  (thêm tham số 600 cho index nhỏ)
python -m advanced.verify_benchmark  # kiểm tra benchmark với PDF gốc
python -m advanced.index_pg          # embed -> pgvector strategy PBB   (thêm 600 cho PBB600)
python -m advanced.run_eval retrieval --split dev --methods dense bm25 hybrid --top-k 4
python -m advanced.run_eval agent --split dev --config C2_hybrid_plan_guard
python -m advanced.review_view --split dev --config C2_hybrid_plan_guard   # hỗ trợ chấm tay
python -m advanced.run_eval summarize --split dev --config C2_hybrid_plan_guard
```

Split test chỉ chạy được sau khi đã có `results/selection.json` (cấu hình được chốt trên dev).

Vòng 2:

```bash
python -m advanced.run_eval agent --split tune --config S_R_scaled_dedupe       # dev + test vòng 1 (49 câu)
python -m advanced.benchmark.sample_test2                                       # rút mục tiêu test2 (seed cố định)
python -m advanced.benchmark.build_test2                                        # ghi benchmark/test2.json
python -m advanced.verify_benchmark --test2 [--freeze]                          # kiểm tra với PDF gốc, khoá
python -m advanced.run_eval agent --split test2 --config S_R_scaled_dedupe      # cần selection_r2.json + FROZEN_test2.json
python -m advanced.review_view --split test2 --config S_R_scaled_dedupe
python -m advanced.run_eval summarize --split test2 --config S_R_scaled_dedupe
```
