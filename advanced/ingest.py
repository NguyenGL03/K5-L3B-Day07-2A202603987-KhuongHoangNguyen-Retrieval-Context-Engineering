"""Index the whole Public Bank 2025 Annual Report (Financial Statements volume).

Digital-born PDF (InDesign), so text and vector graphics are read from the PDF
objects with PyMuPDF instead of OCR:

* text + tables : spans are re-assembled into visual rows by coordinates; column
  gaps become " | ". Table header rows (Group/Bank, 2025/2024, RM'000 ...) are
  carried into every chunk of the same page so a chunk never loses its columns.
  Pages whose text is rotated 90 degrees (landscape tables) are de-rotated.
* bar charts    : year axis labels + rotated value labels, matched by x position.
* donut charts  : each % label -> nearest ring segment (vector path) -> its fill
  colour -> nearest legend swatch colour -> legend text.
* cycle diagrams: curved per-character stage labels are re-joined; bullet lists
  are assigned to the nearest stage; bold oversight boxes are listed separately.

Every visual gets a PNG crop + bbox so its chunk can be checked against the
original. Gold answers / benchmark never enter this module.

Run: python -m advanced.ingest
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
PDF = ROOT.parent / "docs" / "publicbank_fr-2025_hr.pdf"
OUT = ROOT / "data"
CROPS = OUT / "crops"
SOURCE_ID = "pbb-ar2025-fs"
SOURCE = "Public Bank Berhad 2025 Integrated Annual Report — Financial Statements"
MAX_CHARS = 1200
YEARS = ["2021", "2022", "2023", "2024", "2025"]
NUM = re.compile(r"^\(?-?[\d,]+(\.\d+)?%?\)?[#*]*$")


def printed(pno: int) -> int | None:
    return pno - 2 if pno >= 3 else None


# ------------------------------------------------------------------ spans
def spans(page) -> list[dict]:
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                t = s["text"]
                if not t.strip():
                    continue
                out.append({"text": t, "bbox": tuple(s["bbox"]), "dir": tuple(round(v, 2) for v in l["dir"]),
                            "size": s["size"], "bold": "Bold" in s["font"] or bool(s["flags"] & 16),
                            "color": s["color"]})
    return out


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(b, r, pad=0):
    cx, cy = center(b)
    return r[0] - pad <= cx <= r[2] + pad and r[1] - pad <= cy <= r[3] + pad


def rows_from_spans(sp: list[dict], rotated: int = 0) -> list[dict]:
    """Group spans into visual rows. rotated: 0, or +1/-1 for 90-degree text."""
    items = []
    for s in sp:
        x0, y0, x1, y1 = s["bbox"]
        if rotated == -1:      # text runs bottom->top: reading x = -y, row = x
            x0, x1, y0, y1 = -y1, -y0, x0, x1
        elif rotated == 1:     # top->bottom
            x0, x1, y0, y1 = y0, y1, -x1, -x0
        items.append({**s, "x0": x0, "x1": x1, "yc": (y0 + y1) / 2, "h": max(1.0, y1 - y0)})
    items.sort(key=lambda s: (s["yc"], s["x0"]))
    rows: list[list[dict]] = []
    for s in items:
        if rows and abs(rows[-1][0]["yc"] - s["yc"]) <= max(2.5, 0.35 * s["h"]):
            rows[-1].append(s)
        else:
            rows.append([s])
    out = []
    for r in rows:
        r.sort(key=lambda s: s["x0"])
        text, prev = "", None
        cells: list[dict] = []          # " | "-separated cells with their x extent (for column labelling)
        for s in r:
            t = s["text"].strip()
            if prev is None:
                text = t
                cells.append({"text": t, "x0": s["x0"], "x1": s["x1"]})
            else:
                gap = s["x0"] - prev["x1"]
                sep = " | " if gap > 6 else ("" if gap < 0.8 and not prev["text"].endswith(" ") else " ")
                text += sep + t
                if sep == " | ":
                    cells.append({"text": t, "x0": s["x0"], "x1": s["x1"]})
                else:
                    cells[-1]["text"] += sep + t
                    cells[-1]["x1"] = s["x1"]
            prev = s
        for c in cells:
            c["text"] = re.sub(r"\s+", " ", c["text"]).strip()
        out.append({"text": re.sub(r"\s+", " ", text).strip(), "y": r[0]["yc"], "x0": r[0]["x0"],
                    "size": max(s["size"] for s in r), "bold": all(s["bold"] for s in r),
                    "h": max(s["h"] for s in r), "cells": [c for c in cells if c["text"]]})
    return [o for o in out if o["text"]]


# ------------------------------------------------------------------ visuals
def bar_charts(page, sp) -> list[dict]:
    hz = [s for s in sp if s["dir"] == (1.0, 0.0)]
    years = [s for s in hz if s["text"].strip() in YEARS]
    groups = defaultdict(list)
    for s in years:
        groups[round(center(s["bbox"])[1] / 4)].append(s)
    charts = []
    for row in groups.values():
        row.sort(key=lambda s: s["bbox"][0])
        i = 0
        while i + 5 <= len(row):
            seq = row[i:i + 5]
            if [s["text"].strip() for s in seq] == YEARS:
                charts.append(seq)
                i += 5
            else:
                i += 1
    out = []
    for seq in charts:
        xs = [center(s["bbox"])[0] for s in seq]
        step = (xs[-1] - xs[0]) / 4
        x_lo, x_hi, y_axis = xs[0] - step, xs[-1] + step, seq[0]["bbox"][1]
        vert = [s for s in sp if s["dir"] in ((0.0, -1.0), (0.0, 1.0))
                and x_lo <= center(s["bbox"])[0] <= x_hi and y_axis - 260 < s["bbox"][1] < y_axis]
        titles = [s for s in hz if s["bold"] and s["text"].strip().isupper() and x_lo - 10 <= s["bbox"][0] <= x_hi
                  and y_axis - 330 < s["bbox"][1] < y_axis - 40 and s["text"].strip() not in YEARS]
        if not vert or not titles:
            continue
        ty = min(s["bbox"][1] for s in titles)
        titles = [s for s in titles if s["bbox"][1] - ty < 25]
        title = " ".join(s["text"].strip() for s in sorted(titles, key=lambda s: (s["bbox"][1], s["bbox"][0])))
        t_bottom = max(s["bbox"][3] for s in titles)
        unit = [s for s in hz if not s["bold"] and x_lo - 10 <= s["bbox"][0] <= x_hi and 0 <= s["bbox"][1] - t_bottom < 12]
        unit = unit[0]["text"].strip() if unit else ""
        vals = defaultdict(str)
        for s in sorted(vert, key=lambda s: -s["bbox"][3]):
            k = min(range(5), key=lambda j: abs(center(s["bbox"])[0] - xs[j]))
            vals[k] += s["text"].replace(" ", "")
        bbox = (x_lo - 5, ty - 5, x_hi + 5, seq[0]["bbox"][3] + 8)
        series = "; ".join(f"{YEARS[j]}: {vals.get(j, '?')}" for j in range(5))
        text = (f"Bar chart '{title}' ({unit}), five-year series 2021-2025 shown as bars with value labels: {series}. "
                f"2025 bar highlighted in red.")
        out.append({"kind": "chart_bar", "title": f"{title} ({unit})", "bbox": bbox, "text": text,
                    "data": {YEARS[j]: vals.get(j) for j in range(5)}})
    return out


def _dist_pt_seg(p, a, b):
    ax, ay, bx, by = a.x, a.y, b.x, b.y
    dx, dy = bx - ax, by - ay
    t = 0 if dx == dy == 0 else max(0, min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _path_dist(p, d):
    best = 1e9
    for it in d["items"]:
        pts = [x for x in it[1:] if isinstance(x, pymupdf.Point)]
        if it[0] == "re":
            r = it[1]
            pts = [r.tl, r.tr, r.br, r.bl, r.tl]
        for a, b in zip(pts, pts[1:]):
            best = min(best, _dist_pt_seg(p, a, b))
    return best


def donut_charts(page, sp) -> list[dict]:
    hz = [s for s in sp if s["dir"] == (1.0, 0.0)]
    centers = [s for s in hz if s["size"] > 12 and s["text"].strip() in YEARS]
    if len(centers) < 2:
        return []
    drawings = page.get_drawings()
    fills = [d for d in drawings if d.get("fill") and max(d["rect"].width, d["rect"].height) > 12
             and min(d["rect"].width, d["rect"].height) > 1.5
             and d["fill"] != (1.0, 1.0, 1.0) and d["rect"].width < 250]
    swatches = [d for d in drawings if d.get("fill") and 5 < d["rect"].width < 12 and 5 < d["rect"].height < 12]
    # section labels (red pills) split the page into horizontal bands
    pills = [d["rect"] for d in drawings if d.get("fill") and d["fill"][0] > 0.8 and d["fill"][1] < 0.2
             and d["rect"].width > 40 and d["rect"].height < 20]
    bands = []
    for r in sorted(pills, key=lambda r: r.y0):
        label = " ".join(s["text"].strip() for s in hz if inside(s["bbox"], r, 1))
        if label:
            bands.append((r.y0, label))
    if not bands:
        bands = [(0, "")]
    labels = [s for s in hz if re.fullmatch(r"\(?-?\d+(\.\d+)?%\)?", s["text"].strip())]
    text_rows = rows_from_spans([s for s in hz if s["size"] < 10])
    out = []
    for c in centers:
        cx, cy = center(c["bbox"])
        band_i = max(i for i, (y, _) in enumerate(bands) if y <= cy) if any(y <= cy for y, _ in bands) else 0
        y_top = bands[band_i][0]
        y_bot = bands[band_i + 1][0] if band_i + 1 < len(bands) else page.rect.y1
        segs = [d for d in fills if math.hypot(center(tuple(d["rect"]))[0] - cx, center(tuple(d["rect"]))[1] - cy) < 110
                and d["rect"].width < 220]
        my_sw = [d for d in swatches if y_top <= d["rect"].y0 < y_bot]
        legend = []
        for sw in my_sw:
            r = sw["rect"]
            words = [s for s in hz if -2 < s["bbox"][0] - r.x1 < 12 and abs(center(s["bbox"])[1] - (r.y0 + r.y1) / 2) < 5]
            if not words:
                continue
            first = words[0]
            txt = first["text"].strip()
            nxt = [s for s in hz if abs(s["bbox"][0] - first["bbox"][0]) < 8 and 0 < s["bbox"][1] - first["bbox"][1] < 13
                   and not any(abs(o["rect"].y0 - s["bbox"][1]) < 6 and abs(o["rect"].x0 - r.x0) < 3 for o in my_sw)]
            if nxt:
                txt += " " + nxt[0]["text"].strip()
            legend.append((sw["fill"], re.sub(r"[\x00-\x1f]", "", txt).strip()))
        if not segs or not legend:
            continue
        cpts = [center(c2["bbox"]) for c2 in centers]
        my_labels = [s for s in labels
                     if min(cpts, key=lambda q: math.hypot(center(s["bbox"])[0] - q[0], center(s["bbox"])[1] - q[1])) == (cx, cy)
                     and math.hypot(center(s["bbox"])[0] - cx, center(s["bbox"])[1] - cy) < 160]

        def ang(p):
            return math.atan2(p[1] - cy, p[0] - cx)

        def adiff(a, b):
            return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

        # segment direction = circular mean angle of its outer-arc points (slices may be drawn
        # as wedges to the centre and masked by a white disc, so only far points count)
        seg_pts = [[p for it in dseg["items"] for p in it[1:] if isinstance(p, pymupdf.Point)] for dseg in segs]
        r_max = max(math.hypot(p.x - cx, p.y - cy) for pts in seg_pts for p in pts)
        seg_ang, background = [], []
        for pts in seg_pts:
            far = [p for p in pts if math.hypot(p.x - cx, p.y - cy) >= 0.75 * r_max] or pts
            a = [ang((p.x, p.y)) for p in far]
            bins = {int((x + math.pi) / (2 * math.pi) * 36) % 36 for x in a}
            background.append(len(bins) >= 30)          # full disc drawn under the other slices
            seg_ang.append(math.atan2(sum(math.sin(x) for x in a), sum(math.cos(x) for x in a)))
        fg = [j for j in range(len(segs)) if not background[j]]
        if len(fg) < len(segs) and fg:                 # background slice = middle of the largest free gap
            mids = sorted(seg_ang[j] % (2 * math.pi) for j in fg)
            gaps = [((mids[(i + 1) % len(mids)] - mids[i]) % (2 * math.pi), mids[i]) for i in range(len(mids))]
            g, start = max(gaps)
            for j in range(len(segs)):
                if background[j]:
                    seg_ang[j] = start + g / 2

        def legend_of(col):
            return min(legend, key=lambda l: sum((a - b) ** 2 for a, b in zip(l[0], col)))[1]

        # Labels keep the circular order of their slices even when nudged apart, so labels and
        # slices are aligned as two clockwise sequences (DP over every rotation, min total angle
        # error). A negative value (a loss) has no slice -> legend entry without a slice.
        pos = [l for l in my_labels if not l["text"].strip().startswith("(")]
        neg = [l for l in my_labels if l["text"].strip().startswith("(")]
        la = sorted(((ang(center(l["bbox"])) % (2 * math.pi), l) for l in pos), key=lambda x: x[0])
        sg = sorted(((seg_ang[j] % (2 * math.pi), j) for j in range(len(segs))), key=lambda x: x[0])
        k, m = len(la), len(sg)
        best = (float("inf"), [])
        for rot in range(m):
            S = sg[rot:] + sg[:rot]
            INF = float("inf")
            dp = [[(INF, None)] * (m + 1) for _ in range(k + 1)]
            dp[0] = [(0.0, None)] * (m + 1)
            for i in range(1, k + 1):
                for j in range(1, m + 1):
                    skip = dp[i][j - 1]
                    take = (dp[i - 1][j - 1][0] + adiff(la[i - 1][0], S[j - 1][0]), (i - 1, j - 1))
                    dp[i][j] = min(skip, take, key=lambda x: x[0]) if skip[0] != INF or take[0] != INF else (INF, None)
            if dp[k][m][0] < best[0]:
                match, i, j = [], k, m
                while i > 0 and j > 0:
                    if dp[i][j] == dp[i][j - 1]:
                        j -= 1
                        continue
                    match.append((i - 1, S[j - 1][1]))
                    i, j = i - 1, j - 1
                best = (dp[k][m][0], match)
        pairs = [(legend_of(segs[j]["fill"]), la[i][1]["text"].strip()) for i, j in best[1]]
        matched = {n for n, _ in pairs}
        missing = [n for _, n in legend if n not in matched]
        for l, n in zip(neg, missing):
            pairs.append((n, l["text"].strip() + " (negative, no segment drawn)"))
        order = {name: i for i, (_, name) in enumerate(legend)}
        pairs.sort(key=lambda p: order.get(p[0], 99))
        title = bands[band_i][1]
        year = c["text"].strip()
        bbox = (min(d["rect"].x0 for d in segs) - 40, min(d["rect"].y0 for d in segs) - 25,
                max(d["rect"].x1 for d in segs) + 45, max(d["rect"].y1 for d in segs) + 15)
        series = "; ".join(f"{n}: {v}" for n, v in pairs)
        text = (f"Donut chart '{title} {year}' (share of total, %), segments labelled with percentages "
                f"(segment matched to legend by colour): {series}.")
        out.append({"kind": "chart_donut", "title": f"{title} {year}", "bbox": bbox, "text": text,
                    "data": dict(pairs), "legend_rows": len(legend)})
    return out


def cycle_diagram(page, sp) -> list[dict]:
    curved = [s for s in sp if s["dir"] not in ((1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0))]
    if len(curved) < 15:
        return []
    clusters: list[list[dict]] = []
    for s in curved:                                     # stream order = reading order along the arc
        if clusters and min(math.hypot(center(s["bbox"])[0] - center(o["bbox"])[0],
                                       center(s["bbox"])[1] - center(o["bbox"])[1]) for o in clusters[-1]) < 16:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    def join_chars(cl):
        # curved labels are one glyph per span and carry no space glyphs; labels are
        # Title Case, so word breaks are restored at lower->Upper boundaries and around '&'
        out = "".join(s["text"] for s in cl)
        out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
        out = re.sub(r"\s*&\s*", " & ", out)
        return re.sub(r"\s+", " ", out).strip()

    raw = []
    for cl in clusters:
        xs = [center(s["bbox"]) for s in cl]
        raw.append([join_chars(cl), (sum(p[0] for p in xs) / len(xs), sum(p[1] for p in xs) / len(xs))])
    stages: dict[str, tuple] = {}
    for name, c in raw:
        near = next((k for k, v in stages.items() if math.hypot(v[0] - c[0], v[1] - c[1]) < 40), None)
        if near is None:
            stages[name] = c
        elif name.replace(" ", "") in near.replace(" ", ""):
            continue                                     # duplicate copy (shadow) of the same label
        else:                                            # 2nd line of a 2-line label
            v = stages.pop(near)
            stages[f"{near} {name}"] = ((v[0] + c[0]) / 2, (v[1] + c[1]) / 2)
    for k in list(stages):                               # "X X" left by a shadow copy -> "X"
        h = k.split(" ")
        if len(h) % 2 == 0 and h[: len(h) // 2] == h[len(h) // 2:]:
            stages[" ".join(h[: len(h) // 2])] = stages.pop(k)
    if len(stages) < 3:
        return []
    # clockwise order starting from the stage closest to 12 o'clock
    sx = sum(v[0] for v in stages.values()) / len(stages)
    sy = sum(v[1] for v in stages.values()) / len(stages)
    clock = lambda v: (math.atan2(v[1] - sy, v[0] - sx) + math.pi / 2) % (2 * math.pi)  # noqa: E731
    stages = dict(sorted(stages.items(), key=lambda kv: clock(kv[1])))
    bullets = [d["rect"] for d in page.get_drawings() if d.get("fill") and d["rect"].width < 8 and d["rect"].height < 8]
    hz = [s for s in sp if s["dir"] == (1.0, 0.0)]
    stage_box = (min(v[0] for v in stages.values()) - 30, min(v[1] for v in stages.values()) - 30,
                 max(v[0] for v in stages.values()) + 30, max(v[1] for v in stages.values()) + 30)
    title = " ".join(s["text"].strip() for s in hz if s["size"] > 10 and inside(s["bbox"], stage_box))
    ys = [v[1] for v in stages.values()]
    region = (40, min(ys) - 110, page.rect.x1 - 40, max(ys) + 140)
    body = [s for s in hz if inside(s["bbox"], region) and not inside(s["bbox"], stage_box) and s["size"] < 10
            and s["bbox"][2] - s["bbox"][0] < 230]           # drop full-width paragraph lines
    mid = page.rect.x1 / 2
    items = []
    for side in (0, 1):                                  # left / right of the cycle, never mixed in one row
        ls = rows_from_spans([s for s in body if not s["bold"] and (center(s["bbox"])[0] > mid) == bool(side)])
        if not ls:
            continue
        indent = min(r["x0"] for r in ls)
        cur = None
        for r in sorted(ls, key=lambda r: r["y"]):
            has_bullet = any(abs((b.y0 + b.y1) / 2 - r["y"]) < 4 and 0 < r["x0"] - b.x1 < 14 for b in bullets)
            if bullets:
                cont = cur and not has_bullet and r["y"] - cur["y_last"] < 14
            else:
                cont = cur and r["x0"] > indent + 4 and r["y"] - cur["y_last"] < 14
            if cont:
                cur["text"] += " " + r["text"]
                cur["y_last"] = r["y"]
            else:
                cur = {"text": r["text"], "x0": r["x0"], "y": r["y"], "y_last": r["y"], "side": side}
                items.append(cur)
    # a list = consecutive items on one side without a large vertical gap; assign list -> nearest stage
    by_stage = defaultdict(list)
    for side in (0, 1):
        its = sorted((i for i in items if i["side"] == side), key=lambda i: i["y"])
        lists, cur = [], []
        for it in its:
            if cur and it["y"] - cur[-1]["y_last"] > 22:
                lists.append(cur)
                cur = []
            cur.append(it)
        if cur:
            lists.append(cur)
        for lst in lists:
            p = (sum(i["x0"] for i in lst) / len(lst) + 50, sum(i["y"] for i in lst) / len(lst))
            st = min(stages, key=lambda k: math.hypot(stages[k][0] - p[0], stages[k][1] - p[1]))
            by_stage[st].extend(i["text"] for i in lst)
    boxes = rows_from_spans([s for s in body if s["bold"]])
    ov, cur = [], None
    boxes = [r for r in boxes if re.search(r"[A-Z]{3}", r["text"]) and r["text"] == r["text"].upper()]
    for r in sorted(boxes, key=lambda r: (round(r["x0"] / 150), r["y"])):
        if cur and r["y"] - cur["y"] < 14 and abs(r["x0"] - cur["x0"]) < 60:
            cur["text"] += " " + r["text"]
            cur["y"] = r["y"]
        else:
            cur = {"text": r["text"], "x0": r["x0"], "y": r["y"]}
            ov.append(cur)
    def where(p):                                        # position relative to the centre of the cycle
        dx, dy = p[0] - sx, p[1] - sy
        if abs(dx) < 60 and abs(dy) < 60:
            return "centre"
        if abs(dy) >= 1.5 * abs(dx):
            return "top" if dy < 0 else "bottom"
        if abs(dx) >= 1.5 * abs(dy):
            return "left" if dx < 0 else "right"
        return ("top" if dy < 0 else "bottom") + "-" + ("left" if dx < 0 else "right")

    ov_pos = {}
    for o in ov:
        pts = [center(s["bbox"]) for s in body if s["bold"] and s["text"].strip() and s["text"].strip() in o["text"]]
        if pts:
            ov_pos[o["text"]] = where((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    stage_pos = {k: where(v) for k, v in stages.items()}
    parts = [f"Stage '{k}': " + "; ".join(v) for k, v in by_stage.items()]
    text = (f"Cycle diagram '{title}': the risk management process is a cycle of stages "
            f"{' -> '.join(stages)}; activities listed next to each stage. " + " | ".join(parts) +
            ". Oversight/assurance functions around the cycle (arrows into the cycle): " +
            "; ".join(o["text"] for o in ov) + ".")
    all_y = [s["bbox"][1] for s in body] + [s["bbox"][3] for s in body]
    bbox = (50, min(all_y) - 5, page.rect.x1 - 50, max(all_y) + 5)
    return [{"kind": "diagram", "title": title, "bbox": bbox, "text": text,
             "data": {"stages": list(stages), "activities": dict(by_stage), "oversight": [o["text"] for o in ov],
                      "stage_positions": stage_pos, "oversight_positions": ov_pos}}]


def box_flow(page, sp) -> list[dict]:
    """Row of >=3 header boxes (filled, bold title) with a text column under each: a process flow."""
    hz = [s for s in sp if s["dir"] == (1.0, 0.0)]
    heads = []
    for d in page.get_drawings():
        r = d["rect"]
        if d.get("fill") and d["fill"] != (1.0, 1.0, 1.0) and 18 < r.height < 45 and 70 < r.width < 160:
            t = " ".join(s["text"].strip() for s in hz if s["bold"] and inside(s["bbox"], r))
            if t:
                heads.append((r, t))
    rows = defaultdict(list)
    for r, t in heads:
        rows[round(r.y0 / 6)].append((r, t))
    out = []
    for row in rows.values():
        if len(row) < 3:
            continue
        row.sort(key=lambda x: x[0].x0)
        cols = []
        y_end = row[0][0].y1
        for r, t in row:
            body = sorted((s for s in hz if r.x0 - 2 <= center(s["bbox"])[0] <= r.x1 + 2 and r.y1 < s["bbox"][1] < r.y1 + 260
                           and not s["bold"]), key=lambda s: (s["bbox"][1], s["bbox"][0]))
            lines, last = [], None
            for s in body:
                if last is not None and s["bbox"][1] - last > 30:
                    break
                lines.append(s["text"].strip())
                last = s["bbox"][3]
            y_end = max(y_end, last or y_end)
            txt = re.sub(r"\s+", " ", " ".join(lines))
            txt = re.sub(r"\s*[•▸►]\s*", " • ", txt).strip(" •")
            cols.append(f"Step '{t}': {txt}")
        bbox = (row[0][0].x0 - 5, row[0][0].y0 - 5, row[-1][0].x1 + 5, y_end + 5)
        steps = [t for _, t in row]
        text = (f"Process flow diagram (boxes connected by arrows, left to right): {' -> '.join(steps)}. "
                + " | ".join(cols) + ".")
        out.append({"kind": "diagram", "title": " -> ".join(steps), "bbox": bbox, "text": text,
                    "data": {"steps": steps, "columns": cols}})
    return out


# ------------------------------------------------------------------ display annotations
# The indexed chunk `text` is never changed by these (the pgvector embeddings stay valid);
# they only build the `display` text that the agent reads in its context.
K_UNIT = re.compile(r"RM\s?['’]\s?000", re.I)
UNIT_HEAD = re.compile(r"\(\s*(?:RM|USD|HKD)\s?['’]?\s?(?:MILLION|BILLION|000|MIL)\s*\)|\(\s*(?:sen|%|times|RM)\s*\)", re.I)
NUMCELL = re.compile(r"^(\(?-?\d[\d,]*(\.\d+)?%?\)?[#*^]*|[–—-])$")
FOOT_RE = re.compile(r"^(\*{1,3}|#{1,2}|\^{1,2}|†|‡)\s*(?:\|\s*)?(\S.*)$")
MARK_RE = re.compile(r"(?<=[\w%.)’'])(\*{1,3}|#{1,2}|\^{1,2}|†|‡)")


def _is_num(t: str) -> bool:
    # a bare 19xx/20xx is a year header: amounts >= 1,000 are always printed with a thousands separator
    return bool(NUMCELL.match(t.strip())) and not re.fullmatch(r"(19|20)\d\d", t.strip())


def _num_cells(r: dict) -> list[dict]:
    cs = r["cells"]
    return [c for k, c in enumerate(cs) if _is_num(c["text"]) and not (k == 0 and len(cs) > 1 and not _is_num(cs[1]["text"]))]


def _nearest_cell(cells: list[dict], col: dict) -> dict:
    mid = (col["x0"] + col["x1"]) / 2
    return min(cells, key=lambda c: abs((c["x0"] + c["x1"]) / 2 - mid))


def label_rows(rows: list[dict], min_cols: int = 5) -> dict[int, str]:
    """Wide tables (>= min_cols numeric columns): rewrite every data row as 'label: column = value; ...'
    with the column label assembled from the header rows above (multi-line headers stacked per column).
    A 4B model cannot reliably count to the 11th ' | ' cell; it can read 'Hong Kong SAR = 1,182,608'."""
    is_data = [len(_num_cells(r)) >= 2 for r in rows]
    out: dict[int, str] = {}
    i, n = 0, len(rows)
    while i < n:
        if not is_data[i]:
            i += 1
            continue
        # header group: up to 7 non-data rows directly above, stopping at a long text line
        h = i - 1
        head = []
        while h >= 0 and not is_data[h] and len(head) < 7 and all(len(c["text"]) <= 45 for c in rows[h]["cells"]):
            head.insert(0, h)
            h -= 1
        # block body: data rows and wrapped-label rows until a new multi-cell header group starts
        j = i
        while j < n:
            if not is_data[j] and len(rows[j]["cells"]) >= 2 and not any(_is_num(c["text"]) for c in rows[j]["cells"]):
                k = j
                while k < n and not is_data[k]:
                    k += 1
                if k < n and k - j <= 7:
                    break                                    # next header group
            j += 1
        body = range(i, j)
        cells = sorted((c for b in body if is_data[b] for c in _num_cells(rows[b])), key=lambda c: c["x1"])
        cols: list[dict] = []
        for c in cells:                                      # cluster right edges (numbers are right-aligned)
            if cols and c["x1"] - cols[-1]["xs"][-1] < 10:
                cols[-1]["xs"].append(c["x1"])
                cols[-1]["x0"] = min(cols[-1]["x0"], c["x0"])
            else:
                cols.append({"xs": [c["x1"]], "x0": c["x0"]})
        for c in cols:
            c["x1"] = max(c["xs"])
        if len(cols) >= min_cols:
            labels = [[] for _ in cols]
            context: list[str] = []                          # header cells that belong to no single column

            def hits(c):
                out_ = []
                for ci, col in enumerate(cols):
                    ov = min(c["x1"], col["x1"]) - max(c["x0"], col["x0"])
                    mid = (col["x0"] + col["x1"]) / 2
                    if ov >= 0.3 * max(col["x1"] - col["x0"], 4) or c["x0"] <= mid <= c["x1"]:
                        out_.append(ci)
                if not out_ and c["x0"] >= cols[0]["x0"] - 10:   # short label between columns -> nearest column
                    cm = (c["x0"] + c["x1"]) / 2
                    out_ = [min(range(len(cols)), key=lambda q: abs((cols[q]["x0"] + cols[q]["x1"]) / 2 - cm))]
                return out_

            for hr in head:
                hc = rows[hr]["cells"]
                assigned = False
                row_hits = any(hits(c) and len(hits(c)) < 3 for c in hc)
                for c in hc:
                    hs = hits(c)
                    if not hs or len(hs) >= 3:               # left of the columns, or a caption over many columns
                        t = c["text"].rstrip(":").strip()
                        if (row_hits and len(t) <= 20) or (len(hc) == 1 and len(t) <= 12):
                            context.append(t)                # e.g. "Group", "2025", "ASSETS"; not wrapped body lines
                        continue
                    assigned = True
                    for ci in hs:
                        labels[ci].append((hr, c["text"]))
                if not assigned and len(hc) * 2 >= len(cols) and len(hc) > 1:   # e.g. Group | Bank between columns
                    for t in [c["text"] for c in hc]:
                        if t.rstrip(":") in context:
                            context.remove(t.rstrip(":"))
                    for ci, col in enumerate(cols):
                        labels[ci].append((hr, _nearest_cell(hc, col)["text"]))

            def compose(lbls):
                full = [re.sub(r"\s+", " ", " ".join(t for _, t in l)).strip() for l in lbls]
                com = [w for w in dict.fromkeys(full[0].split()) if all(w in l.split() for l in full)] if full else []
                return [" ".join(w for w in l.split() if w not in com) or f"column {k + 1}"
                        for k, l in enumerate(full)], com

            short, common = compose(labels)
            if len(set(short)) < len(short):
                # repeated blocks of columns (2025 | 2024, Group | Bank): a spanning header row centred over
                # each block tells them apart -> give every column the nearest cell of that row instead
                for hr in head:
                    hc = rows[hr]["cells"]
                    if not 2 <= len(hc) < len(cols):
                        continue
                    trial = [[x for x in l if x[0] != hr] + [(hr, _nearest_cell(hc, col)["text"])]
                             for l, col in zip(labels, cols)]
                    t_short, t_common = compose(trial)
                    if len(set(t_short)) == len(t_short):
                        labels, short, common = trial, t_short, t_common
                        context = [x for x in context if x not in {c["text"].rstrip(":").strip() for c in hc}]
                        break
            ctx_words = list(dict.fromkeys(context + common))
            suffix = f" ({', '.join(ctx_words)})" if ctx_words else ""
            for b in body:
                if not is_data[b]:
                    continue
                r = rows[b]
                nums = _num_cells(r)
                lab = " ".join(c["text"] for c in r["cells"] if c not in nums)
                p, pre = b - 1, []                           # wrapped row label lines directly above
                while p >= i and not is_data[p] and len(rows[p]["cells"]) == 1 and len(pre) < 3 \
                        and rows[p + 1]["y"] - rows[p]["y"] < 1.6 * max(rows[p]["h"], 6) and not rows[p]["bold"]:
                    pre.insert(0, rows[p]["cells"][0]["text"])
                    out[p] = ""                              # the wrapped line is repeated in the labelled row
                    p -= 1
                lab = " ".join(pre + [lab]).strip() or "(row)"
                vals = []
                for c in nums:
                    ci = min(range(len(cols)), key=lambda q: abs(cols[q]["x1"] - c["x1"]))
                    vals.append(f"{short[ci]} = {c['text']}")
                sfx = suffix
                if all("%" in c["text"] or c["text"] in "–—-" for c in nums):   # a ratio row is not in RM'000
                    kept_w = [w for w in ctx_words if not K_UNIT.search(w)]
                    sfx = f" ({', '.join(kept_w)})" if kept_w else ""
                out[b] = f"{lab}: " + "; ".join(vals) + sfx
        i = max(j, i + 1)
    return out


def page_footnotes(rows: list[dict]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for k, r in enumerate(rows):
        m = FOOT_RE.match(r["text"])
        if not m or len(m.group(2)) < 8:
            continue
        txt = m.group(2).replace(" | ", " ")
        for nxt in rows[k + 1:k + 3]:
            if FOOT_RE.match(nxt["text"]) or nxt["y"] - r["y"] > 30:
                break
            txt += " " + nxt["text"].replace(" | ", " ")
        notes.setdefault(m.group(1), txt.strip())
    return notes


def visual_display(v: dict, notes: dict[str, str]) -> str:
    extra = []
    if v["kind"].startswith("chart"):
        marks = list(dict.fromkeys(MARK_RE.findall(v["text"])))
        hits = [f"'{m}' = {notes[m]}" for m in marks if m in notes]
        if hits:
            extra.append("Footnotes printed under the charts on this page: " + "; ".join(hits) + ".")
    if v["kind"] == "diagram" and v["data"].get("stage_positions"):
        sp_ = "; ".join(f"{k} ({p})" for k, p in v["data"]["stage_positions"].items())
        op = "; ".join(f"{k} ({p})" for k, p in v["data"].get("oversight_positions", {}).items())
        extra.append(f"Layout of the figure (position relative to the centre of the cycle): stages: {sp_}."
                     + (f" Oversight boxes: {op}." if op else ""))
    return " ".join([v["text"]] + extra)


# ------------------------------------------------------------------ headings
NOTE_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Z0-9 ,’'&()/\-.:]{4,})$")


def clean_heading(t: str) -> str:
    return re.sub(r"\s*\((CONTINUED|CONT['’]D\.?)\)\.?\s*$", "", t.replace(" | ", " ")).strip()


def is_header_row(t: str) -> bool:
    toks = t.split(" | ")
    yearish = sum(bool(re.fullmatch(r"(20\d\d)( RM['’](000|Million))?( \(Restated\))?", x.strip())) for x in toks)
    return ("RM’000" in t or "RM'000" in t or yearish >= 2 or t.strip() in ("Group | Bank", "Group", "Bank")) \
        and not re.search(r"\d{1,3},\d{3}", t)


def build(max_chars: int = MAX_CHARS) -> list[dict]:
    suffix = "" if max_chars == MAX_CHARS else f"_{max_chars}"
    doc = pymupdf.open(PDF)
    CROPS.mkdir(parents=True, exist_ok=True)
    chunks, visuals, skipped = [], [], []
    part, note = "", ""
    for pno in range(1, doc.page_count + 1):
        page = doc[pno - 1]
        sp = spans(page)
        if not sp:
            skipped.append({"page": pno, "reason": "no text layer (blank or full-page image)"})
            continue
        vis = bar_charts(page, sp) + donut_charts(page, sp) + cycle_diagram(page, sp) + box_flow(page, sp)
        dirs = defaultdict(int)
        for s in sp:
            dirs[s["dir"]] += len(s["text"])
        rotated = 0
        main_dir = max(dirs, key=dirs.get)
        if main_dir == (0.0, -1.0):
            rotated = -1
        elif main_dir == (0.0, 1.0):
            rotated = 1
        keep_dir = main_dir
        plain = [s for s in sp if s["dir"] == keep_dir and not any(inside(s["bbox"], v["bbox"]) for v in vis)]
        rows = rows_from_spans(plain, rotated)
        if rotated == 0:
            rows = [r for r in rows if not (r["y"] < 40 or r["y"] > page.rect.y1 - 25)]  # running header/footer
        big = [r["text"] for r in rows if r["size"] >= 18]
        if big:
            part = " ".join(big)
        notes = page_footnotes(rows)
        labelled = label_rows(rows)
        for k, v in enumerate(vis, 1):
            r = pymupdf.Rect(v["bbox"]) & page.rect
            crop = f"p{pno:03d}_{v['kind']}_{k}.png"
            page.get_pixmap(dpi=110, clip=r).save(CROPS / crop)
            cid = f"{SOURCE_ID}-p{pno:03d}-v{k}"
            visuals.append({"id": cid, "page": pno, "kind": v["kind"], "title": v["title"], "data": v["data"]})
            chunks.append({"id": cid, "text": f"[{part}] {v['text']}", "display": f"[{part}] {visual_display(v, notes)}",
                           "source_id": SOURCE_ID, "source": SOURCE,
                           "page": pno, "metadata": {"printed_page": printed(pno), "kind": v["kind"], "section": part,
                                                     "note": note, "title": v["title"], "bbox": [round(x, 1) for x in r],
                                                     "crop": f"crops/{crop}"}})
        header: list[str] = []
        buf: list[str] = []
        dbuf: list[str] = []                     # display lines, parallel to buf
        k = 0
        unit_head = ""                           # last unit-bearing heading on this page, e.g. "... (RM’MILLION)"
        start_unit = ""                          # the heading in effect where the current chunk starts

        def flush():
            nonlocal buf, dbuf, k
            if buf and any(len(x) > 3 for x in buf):
                k += 1
                prefix = f"[{part}{' > ' + note if note else ''}]"
                shown = list(dbuf)
                # a chunk that starts below a unit heading (e.g. 'KEY BALANCE SHEET DATA (RM’MILLION)' in the
                # previous chunk) and states no unit of its own gets that heading repeated in its display
                if start_unit and not any(UNIT_HEAD.search(x) or K_UNIT.search(x) for x in buf[:3]):
                    shown.insert(0, f"(unit heading above this part of the page: {start_unit})")
                chunks.append({"id": f"{SOURCE_ID}-p{pno:03d}-t{k}", "text": prefix + "\n" + "\n".join(buf),
                               "display": prefix + "\n" + "\n".join(shown),
                               "source_id": SOURCE_ID, "source": SOURCE, "page": pno,
                               "metadata": {"printed_page": printed(pno), "kind": "table_text" if rotated == 0 else
                                            "table_rotated", "section": part, "note": note}})
            buf, dbuf = [], []

        prev_header = False
        for ri, r in enumerate(rows):
            t = r["text"]
            m = NOTE_RE.match(clean_heading(t)) if r["bold"] else None
            if m:
                flush()
                note = clean_heading(t)
                header = []
            if is_header_row(t):
                header = (header + [t])[-3:] if prev_header else [t]
                prev_header = True
            else:
                prev_header = False
            if sum(len(x) + 1 for x in buf) + len(t) > max_chars:
                flush()
                if header and not prev_header:
                    buf = ["(columns: " + " / ".join(header) + ")"]
                    dbuf = list(buf)
            if not any(not x.startswith("(columns: ") for x in buf):
                start_unit = unit_head
            buf.append(t)
            if labelled.get(ri, t):
                dbuf.append(labelled.get(ri, t))
            if UNIT_HEAD.search(t) and not re.search(r"\d{1,3},\d{3}|\d\.\d", t):
                unit_head = t
        flush()
    meta = {"pdf": PDF.name, "pages": doc.page_count, "chunks": len(chunks),
            "by_kind": {kd: sum(c["metadata"]["kind"] == kd for c in chunks)
                        for kd in sorted({c["metadata"]["kind"] for c in chunks})},
            "skipped_pages": skipped, "max_chars": max_chars,
            "decorative_images_not_indexed": [{"page": i + 1, "images": len(doc[i].get_images())}
                                              for i in range(doc.page_count) if doc[i].get_images()]}
    OUT.mkdir(exist_ok=True)
    (OUT / f"chunks{suffix}.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (OUT / "visuals.json").write_text(json.dumps(visuals, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (OUT / f"index_stats{suffix}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return chunks


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_CHARS
    build(n)
    print(json.loads((OUT / f"index_stats{'' if n == MAX_CHARS else '_' + str(n)}.json").read_text(encoding="utf-8")))
