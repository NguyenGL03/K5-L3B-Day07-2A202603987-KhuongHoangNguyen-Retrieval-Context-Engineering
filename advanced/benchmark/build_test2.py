"""Writes advanced/benchmark/test2.json: the 30 round-2 test questions, hand-written for the targets drawn by
sample_test2.py (seed 20261116) AFTER results/selection_r2.json was fixed. Every value was read on the original
PDF page (text pages via PyMuPDF text, charts/diagrams on the rendered page); calculations are computed here, not
typed. Checked by: python -m advanced.verify_benchmark --test2

Run: python -m advanced.benchmark.build_test2
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = "visual: crop advanced/data/crops/{} (bbox in chunk metadata), read on the page rendered with PyMuPDF"


def pct(a: float, b: float) -> float:
    return round(a / b * 100, 3)


def build() -> list[dict]:
    m1, m2, m3, m4 = pct(13549481, 29509548), pct(1259611, 333871515), pct(217993, 29509548), pct(375716531, 499209549)
    q: list[dict] = []

    def add(**k):
        q.append({"split": "test2", **k})

    # ------------------------------------------------------------------ table (6)
    add(id="t2_01", group="table",
        question="In the note on deposits from banks and other financial institutions, how much did the Bank (company level) "
                 "hold as deposits from other financial institutions at 31 December 2025, and what was the corresponding Group figure?",
        answer="Bank: RM8,103,039 thousand; Group: RM2,371,188 thousand (31 December 2025).",
        must=["8,103,039 (Bank 2025)", "2,371,188 (Group 2025)"], unit="RM'000", calc=None,
        gold=[{"page": 135, "region": "Note 23, row 'Other financial institutions', Group 2025 and Bank 2025",
               "evidence": "Other financial institutions 2,371,188 4,107,467 8,103,039 9,076,623"}],
        accept_pages=[135], target="pbb-ar2025-fs-p135-t3",
        reason="Tra đúng hàng và phân biệt cột Group/Bank (ở dòng này Bank lớn hơn Group).")
    add(id="t2_02", group="table",
        question="What did the Group report for reverse repurchase agreements in its statement of financial position at "
                 "31 December 2025, and what was the restated 2024 comparative?",
        answer="RM1,642 thousand at 31 December 2025, against RM6,103 thousand (restated) for 2024.",
        must=["1,642 (2025)", "6,103 (2024)"], unit="RM'000", calc=None,
        gold=[{"page": 29, "region": "Statements of Financial Position, 'Reverse repurchase agreements' (Note 4(a)), Group",
               "evidence": "Reverse repurchase agreements 4(a) 1,642 6,103 1,642 6,103"}],
        accept_pages=[29], target="pbb-ar2025-fs-p029-t1",
        reason="Tra một dòng nhỏ trong bảng cân đối; số 2024 là số trình bày lại.")
    add(id="t2_03", group="table",
        question="After deducting dividends declared subsequent to the financial year end, what were the Group's Tier I "
                 "capital ratios for 2025 and 2024?",
        answer="13.909% for 2025 and 14.192% for 2024 (after deducting dividends; before deducting: 14.540% and 14.803%).",
        must=["13.909% (2025)", "14.192% (2024)"], unit="%", calc=None,
        gold=[{"page": 241, "region": "Note 52(a), block 'After deducting dividends*', 'Tier I capital ratio', Group",
               "evidence": "After deducting dividends*: ... Tier I capital ratio 13.909% 14.192% 12.040% 12.245%"}],
        accept_pages=[241], target="pbb-ar2025-fs-p241-t1",
        reason="Có điều kiện: khối 'After deducting dividends', không phải 'Before'.")
    add(id="t2_04", group="table",
        question="In 2025, how much did the Group pay in audit fees to firms affiliated with the parent auditor, and how much "
                 "in audit fees to the parent auditor itself?",
        answer="RM3,032 thousand to firms affiliated with the parent auditor and RM3,944 thousand to the parent auditor (Group, 2025).",
        must=["3,032 (affiliated firms, Group 2025)", "3,944 (parent auditor, Group 2025)"], unit="RM'000", calc=None,
        gold=[{"page": 158, "region": "Note 38, Auditors' remuneration, '– Audit' under 'Parent auditor' and under "
                                     "'Firms affiliated with parent auditor', Group 2025",
               "evidence": "Parent auditor – Audit 3,944 3,754 2,987 2,831 ... Firms affiliated with parent auditor – Audit 3,032 3,171 31 30"}],
        accept_pages=[158], target="pbb-ar2025-fs-p158-t2",
        reason="Hai dòng cùng tên '– Audit' dưới hai nhóm khác nhau.")
    add(id="t2_05", group="table",
        question="What interest expense did the Group record under 'Others' in the interest expense note for 2025, and by "
                 "how much did it change compared with 2024?",
        answer="RM74,963 thousand in 2025, up RM20,584 thousand from RM54,379 thousand in 2024.",
        must=["74,963", "+20,584 (increase)"], unit="RM'000", calc="74,963 - 54,379 = 20,584",
        gold=[{"page": 153, "region": "Note 34 Interest expense, row 'Others', Group 2025 / 2024",
               "evidence": "34. INTEREST EXPENSE ... Others 74,963 54,379 61,785 56,778"}],
        accept_pages=[153], target="pbb-ar2025-fs-p153-t3",
        reason="'Others' cũng có ở Note 33 cùng trang (thu nhập lãi 75,870); phải lấy đúng note và tính chênh lệch.")
    add(id="t2_06", group="table",
        question="In the related party transactions note, what total expenditure did the Bank incur in transactions with its "
                 "subsidiary companies in 2025 and in 2024?",
        answer="RM360,853 thousand in 2025 and RM291,221 thousand in 2024.",
        must=["360,853 (2025)", "291,221 (2024)"], unit="RM'000", calc=None,
        gold=[{"page": 168, "region": "Note 45(a), Bank, 'Expenditure incurred' total row, 'Subsidiary Companies' 2025 / 2024",
               "evidence": "71,035 54,423 360,853 291,221 16,661 13,062"}],
        accept_pages=[168], target="pbb-ar2025-fs-p168-t6",
        reason="Bảng rộng 10 cột (5 nhóm bên liên quan x 2 năm); dòng tổng không có nhãn.")
    # ------------------------------------------------------------------ visual (5)
    add(id="g2_01", group="visual",
        question="In the Risk Appetite framework flow diagram, what is the first step, and which three considerations is the "
                 "desired risk appetite set against in that step?",
        answer="Set Risk Appetite: the desired risk appetite is set considering strategic business directions, risk taking "
               "capacity and the current risk profile (and articulated using Risk Appetite Metrics).",
        must=["Set Risk Appetite", "Strategic business directions", "Risk taking capacity", "Current risk profile"], unit=None, calc=None,
        gold=[{"page": 278, "region": "Risk Appetite flow, first (left) box 'Set Risk Appetite'", "evidence": V.format("p278_diagram_2.png")}],
        accept_pages=[278], target="pbb-ar2025-fs-p278-v2", reason="Phần tử rút theo seed: 'first step and what it considers'.")
    add(id="g2_02", group="visual",
        question="Which activities does the liquidity and funding risk management processes diagram list under the "
                 "'Identification' stage?",
        answer="Review the Liquidity Positions; Review the Contractual and Behavioural Profiles; Identify Abnormalities and "
               "Concentrations; Review of New Products/Strategies for Potential Liquidity Risk; Emerging Risk Identification.",
        must=["Review the Liquidity Positions", "Review the Contractual and Behavioural Profiles",
              "Identify Abnormalities and Concentrations", "Review of New Products/Strategies for Potential Liquidity Risk",
              "Emerging Risk Identification"], unit=None, calc=None,
        gold=[{"page": 322, "region": "Liquidity cycle, list at top-right next to 'Identification'", "evidence": V.format("p322_diagram_1.png")}],
        accept_pages=[322], target="pbb-ar2025-fs-p322-v1", reason="Phần tử rút theo seed: 'Identification activities'.")
    add(id="g2_03", group="visual",
        question="According to the market risk management processes diagram, which activity is strictly prohibited unless "
                 "approval from the Board is obtained, and under which stage of the cycle is it listed?",
        answer="Derivative activities for trading; listed under Control & Mitigation.",
        must=["Derivative activities for trading", "Control & Mitigation"], unit=None, calc=None,
        gold=[{"page": 318, "region": "Market risk cycle, list at bottom-left next to 'Control & Mitigation', last bullet",
               "evidence": V.format("p318_diagram_1.png")}],
        accept_pages=[318], target="pbb-ar2025-fs-p318-v1", reason="Phần tử rút theo seed: 'activity prohibited without Board approval'.")
    add(id="g2_04", group="visual",
        question="In the operational risk management processes diagram, which oversight or assurance function is shown on "
                 "the left side of the cycle?",
        answer="Independent Risk Management & Compliance Review.",
        must=["Independent Risk Management & Compliance Review"], unit=None, calc=None,
        gold=[{"page": 323, "region": "Operational risk cycle, box with arrow into the cycle from the left",
               "evidence": V.format("p323_diagram_1.png")}],
        accept_pages=[323], target="pbb-ar2025-fs-p323-v1", reason="Phần tử rút theo seed: 'oversight box on the left' (vị trí trên trang gốc).")
    add(id="g2_05", group="visual",
        question="Which activities does the Shariah risk management processes diagram list under the 'Monitoring' stage?",
        answer="Periodic Risk Profile Reporting; Key Risk Indicators; Shariah Risk Incident Data Collection; Principal Risk Positions.",
        must=["Periodic Risk Profile Reporting", "Key Risk Indicators", "Shariah Risk Incident Data Collection",
              "Principal Risk Positions"], unit=None, calc=None,
        gold=[{"page": 325, "region": "Shariah risk cycle, list at top-left next to 'Monitoring'", "evidence": V.format("p325_diagram_1.png")}],
        accept_pages=[325], target="pbb-ar2025-fs-p325-v1", reason="Phần tử rút theo seed: 'Monitoring activities'.")
    # ------------------------------------------------------------------ chart (5)
    add(id="c2_01", group="chart",
        question="In the five-year growth chart of the gross impaired loans ratio, in which year was the ratio highest and "
                 "what was it, and what was the 2025 value?",
        answer="Highest in 2023 at 0.59%; 0.51% in 2025.", must=["2023", "0.59%", "0.51% (2025)"], unit="%", calc=None,
        gold=[{"page": 9, "region": "Bar chart 'GROSS IMPAIRED LOANS RATIO', Percentage (%)", "evidence": V.format("p009_chart_bar_4.png")}],
        accept_pages=[9, 7], target="pbb-ar2025-fs-p009-v4", reason="Đọc nhãn giá trị theo năm và tìm cực đại.")
    add(id="c2_02", group="chart",
        question="What share price does the five-year growth chart show for 2025, and was it higher or lower than in 2024?",
        answer="RM4.54 in 2025, slightly lower than RM4.56 in 2024 (down RM0.02).",
        must=["RM4.54 (2025)", "RM4.56 (2024)", "lower"], unit="RM", calc="4.54 - 4.56 = -0.02",
        gold=[{"page": 8, "region": "Bar chart 'SHARE PRICE', RM", "evidence": V.format("p008_chart_bar_5.png")}],
        accept_pages=[8, 6], target="pbb-ar2025-fs-p008-v5", reason="Đọc đơn vị RM, so sánh hai cột gần bằng nhau.")
    add(id="c2_03", group="chart",
        question="In the 2024 total assets segment chart, which business segment had the largest share, and which overseas "
                 "location had the largest share?",
        answer="Retail operations, 50.1%; among overseas locations, Hong Kong SAR, 3.6%.",
        must=["Retail operations 50.1%", "Hong Kong SAR 3.6%"], unit="%", calc=None,
        gold=[{"page": 11, "region": "Donut chart 'TOTAL ASSETS' 2024 and legend", "evidence": V.format("p011_chart_donut_6.png")}],
        accept_pages=[11, 10], target="pbb-ar2025-fs-p011-v6", reason="Ghép lát với chú giải theo màu; tách mảng trong nước/nước ngoài.")
    add(id="c2_04", group="chart",
        question="According to the dividend per share chart, by how many sen did the dividend per share increase from 2021 to 2025?",
        answer="By 7.3 sen, from 15.2 sen in 2021 to 22.5 sen in 2025.",
        must=["15.2 (2021)", "22.5 (2025)", "7.3 sen"], unit="sen", calc="22.5 - 15.2 = 7.3",
        gold=[{"page": 8, "region": "Bar chart 'DIVIDEND PER SHARE', Sen", "evidence": V.format("p008_chart_bar_4.png")}],
        accept_pages=[8, 6], target="pbb-ar2025-fs-p008-v4", reason="Đọc hai đầu chuỗi năm, đơn vị sen, tính chênh lệch.")
    add(id="c2_05", group="chart",
        question="In the 2024 'Liabilities & Equity' chart of the Simplified Group Balance Sheet, what shares did deposits "
                 "from banks and other financial institutions and debt securities issued and other borrowed funds represent?",
        answer="Deposits from banks and other financial institutions 2.5%; debt securities issued and other borrowed funds 2.0%.",
        must=["2.5% (deposits from banks and other FIs)", "2.0% (debt securities issued and other borrowed funds)"], unit="%", calc=None,
        gold=[{"page": 5, "region": "Donut chart 'LIABILITIES & EQUITY' 2024", "evidence": V.format("p005_chart_donut_2.png")}],
        accept_pages=[5], target="pbb-ar2025-fs-p005-v2", reason="Hai lát nhỏ cạnh nhau; ghép nhãn % với chú giải theo màu.")
    # ------------------------------------------------------------------ multihop (4)
    add(id="m2_01", group="multihop",
        question="Total value distributed in the 2025 Statement of Value Added was what percentage of the Group's 2025 operating revenue?",
        answer=f"About {m1:.1f}% (RM13,549,481 thousand / RM29,509,548 thousand).",
        must=[f"≈{m1:.1f}%", "13,549,481", "29,509,548 (or RM29,510 million)"], unit="%",
        calc=f"13,549,481 / 29,509,548 x 100 = {m1}%",
        gold=[{"page": 12, "region": "Statement of Value Added, 'Total distributed' 2025", "evidence": "Total distributed 13,549,481 12,814,094"},
              {"page": 31, "region": "Statements of Profit or Loss, Operating revenue, Group 2025", "evidence": "Operating revenue ... 29,509,548"}],
        accept_pages=[12, 31, 247, 4], required_pages=[[12], [31, 247, 4]], target="pbb-ar2025-fs-p012-t3",
        reason="Neo rút theo seed: 'Total distributed' (p12); bước 2 = doanh thu hoạt động ở trang khác.")
    add(id="m2_02", group="multihop",
        question="The Financial Highlights report the Bank's (company) gross impaired loans ratio for 2025. Using the Bank's "
                 "impaired loans balance at 31 December 2025 and its gross loans, advances and financing from the loans note, "
                 "check that ratio.",
        answer=f"Financial Highlights: 0.38% (Bank, 2025). Loans note: RM1,259,611 thousand / RM333,871,515 thousand = "
               f"{m2:.3f}%, about 0.38%, so the ratio is consistent.",
        must=["0.38% (Highlights)", "1,259,611", "333,871,515", f"≈{m2:.2f}%, consistent with 0.38%"], unit="%",
        calc=f"1,259,611 / 333,871,515 x 100 = {m2}%",
        gold=[{"page": 4, "region": "Financial Highlights, 'Gross impaired loans ratio', Bank 2025",
               "evidence": "Gross impaired loans ratio"},
              {"page": 88, "region": "Note 10, Bank 2025: gross loans total and impaired loans 'At 31 December'",
               "evidence": "445,758,078 424,170,971 333,871,515 316,795,258 ... At 31 December 2,284,573 2,225,493 1,259,611 1,243,812"}],
        accept_pages=[4, 88], required_pages=[[4, 88], [88]], target="pbb-ar2025-fs-p004-t3",
        reason="Neo rút theo seed: tỷ lệ nợ xấu (p4); bước 2 = số dư nợ xấu và tổng dư nợ của Bank (Note 10, p88).")
    add(id="m2_03", group="multihop",
        question="What share of 2025 Group operating revenue does the segmental analysis give for investment banking, what "
                 "external revenue does the segment information note report for investment banking in 2025, and what share "
                 "of the Group's total external revenue does that amount represent?",
        answer=f"0.7% in the segmental analysis; external revenue RM217,993 thousand in Note 54; that is {m3:.2f}% of the "
               f"Group total of RM29,509,548 thousand (consistent with 0.7%).",
        must=["0.7% (segmental analysis)", "217,993", f"≈{m3:.2f}%"], unit="%", calc=f"217,993 / 29,509,548 x 100 = {m3}%",
        gold=[{"page": 10, "region": "Segmental Analysis, Investment banking, Operating Revenue 2025 (%)",
               "evidence": "Investment banking 0.7 0.9 0.7 1.0 0.4 0.4"},
              {"page": 247, "region": "Note 54 (rotated), External revenue, column 'Investment Banking'",
               "evidence": "External revenue 3,892,661 13,592,846 2,177,646 1,423,632 217,993"}],
        accept_pages=[10, 11, 247], required_pages=[[10, 11], [247]], target="pbb-ar2025-fs-p010-t1",
        reason="Neo rút theo seed: dòng Investment banking (p10); bước 2 = bảng xoay Note 54 (p247).")
    add(id="m2_04", group="multihop",
        question="The Five-Year Group Financial Summary gives the Group's 2025 total liabilities. What were the Bank's "
                 "(company) total liabilities at 31 December 2025 in the Statements of Financial Position, and what share "
                 "of the Group's total liabilities do they represent?",
        answer=f"Group total liabilities RM499,210 million (five-year summary); Bank total liabilities RM375,716,531 "
               f"thousand (statement of financial position), about {m4:.1f}% of the Group's.",
        must=["499,210 million (or 499,209,549 thousand)", "375,716,531", f"≈{m4:.1f}%"], unit="%",
        calc=f"375,716,531 / 499,209,549 x 100 = {m4}%",
        gold=[{"page": 6, "region": "Five-Year Group Financial Summary, Total liabilities 2025 (RM'Million)",
               "evidence": "Total liabilities 499,210 483,380*"},
              {"page": 29, "region": "Statements of Financial Position, TOTAL LIABILITIES, Bank 2025",
               "evidence": "TOTAL LIABILITIES 499,209,549 483,380,277 375,716,531 361,922,515"}],
        accept_pages=[6, 29], required_pages=[[6, 29], [29]], target="pbb-ar2025-fs-p006-t2",
        reason="Neo rút theo seed: Total liabilities (p6); bước 2 = cột Bank ở bảng cân đối (p29); số Group ở p29 cũng chấp nhận.")
    # ------------------------------------------------------------------ text (4)
    add(id="x2_01", group="text",
        question="According to the Corporate Philosophy, in what two ways does Public Bank say it cares for its employees?",
        answer="By promoting staff well-being through attractive remuneration and fringe benefits, and by promoting good "
               "staff morale through proper staff training and development and the provision of opportunities for career advancement.",
        must=["well-being through attractive remuneration and fringe benefits",
              "staff morale through training and development and career advancement opportunities"], unit=None, calc=None,
        gold=[{"page": 3, "region": "Corporate Philosophy, 'For Its Employees'",
               "evidence": "By promoting the well-being of its staff through attractive remuneration and fringe benefits ... "
                           "By promoting good staff morale through proper staff training and development and provision of "
                           "opportunities for career advancement"}],
        accept_pages=[3], target="page 3", reason="Điều kiện: chỉ phần 'For Its Employees'.")
    add(id="x2_02", group="text",
        question="For cash flow hedges, what gain or loss on the ineffective portions did the Group and the Bank recognise "
                 "in 2025, under which line, and what was the 2024 figure?",
        answer="A gain of RM1,861,000 in 2025 (2024: a loss of RM2,043,000), recognised in profit or loss under 'Net gains "
               "and losses on financial instruments' (Note 36).",
        must=["gain of RM1,861,000 (2025)", "loss of RM2,043,000 (2024)", "Net gains and losses on financial instruments"],
        unit="RM", calc=None,
        gold=[{"page": 75, "region": "Note 6, Cash Flow Hedge, last paragraph",
               "evidence": "under “Net gains and losses on financial instruments”. During the financial year, a gain of "
                           "RM1,861,000 (2024 – loss of RM2,043,000) (Note 36) was recognised by the Group and the Bank."}],
        accept_pages=[75], target="page 75",
        reason="Điều kiện: phòng ngừa dòng tiền (không phải fair value hedge ở p74); lãi/lỗ đổi chiều giữa hai năm.")
    add(id="x2_03", group="text",
        question="For the Bank's subsidiaries that are not wholly owned, what rights do non-controlling shareholders hold, "
                 "and are these subsidiaries material to the Group?",
        answer="Non-controlling shareholders hold protective rights restricting the Bank's ability to use the subsidiaries' "
               "assets and settle the Group's liabilities unless their approval is obtained; the subsidiaries with "
               "non-controlling interests are not material, individually or in aggregate, to the Group.",
        must=["protective rights restricting use of assets / settling liabilities unless approved",
              "not material individually or in aggregate"], unit=None, calc=None,
        gold=[{"page": 120, "region": "Note 16, paragraphs after the PBSVN acquisition table",
               "evidence": "non-controlling shareholders hold protective rights restricting the Bank’s ability to use the "
                           "assets of the subsidiary companies and settle the liabilities of the Group ... are not material "
                           "individually or in aggregate"}],
        accept_pages=[120], target="page 120", reason="Tổng hợp hai ý có điều kiện (công ty con không sở hữu toàn bộ).")
    add(id="x2_04", group="text",
        question="Which director was appointed to the board of Public Bank Securities Vietnam Company Limited on 1 January "
                 "2026, and who retired from that board on the same date?",
        answer="Mr Chee Keng Eng was appointed and Dato' Chang Kat Kiam retired, both on 1 January 2026.",
        must=["Mr Chee Keng Eng (appointed)", "Dato' Chang Kat Kiam (retired)"], unit=None, calc=None,
        gold=[{"page": 274, "region": "Appendix, Directors of subsidiary companies, 'Public Bank Securities Vietnam Company Limited'",
               "evidence": "Mr Chee Keng Eng (appointed on 1 January 2026) Dato’ Chang Kat Kiam (retired on 1 January 2026)"}],
        accept_pages=[274], target="page 274", reason="Điều kiện theo công ty con và ngày; tên người lặp lại ở nhiều công ty con.")

    # ------------------------------------------------------------------ unanswerable (6)
    def un(i, question, pats, why):
        add(id=i, group="unanswerable", question=question, expected_behavior="abstain", absence_check=pats, reason=why)

    un("u2_01", "What was the Group's average cost of funds in 2025?",
       [r"cost of funds", r"funding costs? (was|were|of|at) \d"],
       "Gần chủ đề (p21 'optimising funding costs', có chi phí lãi) nhưng không có chỉ số cost of funds; tự tính là suy đoán.")
    un("u2_02", "How many cyber incidents or data breaches did Public Bank experience in 2025?",
       [r"cyber ?attack", r"data breach", r"cyber incident"],
       "Gần chủ đề (p20, p220 nói về cyber security) nhưng không có số sự cố.")
    un("u2_03", "How many training hours per employee did the Public Bank Group provide in 2025?",
       [r"training hours", r"hours of training", r"learning hours"],
       "Gần chủ đề (nhân sự, đào tạo ở p3) nhưng không có số giờ đào tạo.")
    un("u2_04", "How many calls did Public Bank's customer contact centre handle in 2025?",
       [r"call cent", r"contact cent"], "Gần chủ đề dịch vụ khách hàng; không có trung tâm liên lạc/số cuộc gọi.")
    un("u2_05", "What loan growth target has the Group set for 2026?",
       [r"loans? growth", r"growth target", r"target(ed)? (loan|financing) growth", r"financing growth"],
       "Gần chủ đề (Business Outlook 2026, p21) nhưng không có mục tiêu tăng trưởng cho vay bằng số.")
    un("u2_06", "What was Public Bank's ranking in The Banker's Top 1000 World Banks 2025 list?",
       [r"The Banker", r"Top 1000"], "Gần chủ đề (p8 có xếp hạng vốn hoá trên Bursa) nhưng không có xếp hạng The Banker.")
    return q


def main() -> None:
    qs = build()
    out = {"document": {"source_id": "pbb-ar2025-fs", "file": "docs/publicbank_fr-2025_hr.pdf",
                        "page_convention": "page = PDF page index (1-based); printed page = page - 2"},
           "round": 2, "written_after": "results/selection_r2.json (configuration selected on `tune`)",
           "targets": "benchmark/test2_targets.json (seed 20261116; diagram elements seed 20261117)",
           "groups": {"table": "Bảng", "visual": "Ảnh, sơ đồ", "chart": "Biểu đồ", "multihop": "Multi-hop",
                      "text": "Text", "unanswerable": "Thiếu bằng chứng"},
           "questions": qs}
    (ROOT / "benchmark" / "test2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(len(qs), "questions")


if __name__ == "__main__":
    main()
