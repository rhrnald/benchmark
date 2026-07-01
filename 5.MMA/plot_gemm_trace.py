#!/usr/bin/env python3
import argparse
import csv
import html


LANES = [
    ("tma_issue", 0, "TMA producer w0: A[256x64] + B0[64x128]"),
    ("tma_issue", 1, "TMA producer w1: B1[64x128]"),
    ("tma_wait", 2, "pipe0 ready wait w2: A + B0"),
    ("tma_wait", 3, "pipe1 ready wait w3: A + B1"),
    ("mma_issue", 2, "MMA pipe0 w2: C00/C10"),
    ("mma_issue", 3, "MMA pipe1 w3: C01/C11"),
    ("mma_wait", 2, "MMA pipe0 completion wait w2"),
    ("mma_wait", 3, "MMA pipe1 completion wait w3"),
    ("tmem_drain", 0, "TMEM drain w0: C00"),
    ("tmem_drain", 1, "TMEM drain w1: C01"),
    ("tmem_drain", 2, "TMEM drain w2: C10"),
    ("tmem_drain", 3, "TMEM drain w3: C11"),
]

COLORS = {
    "tma_issue": "#087f5b",
    "tma_wait": "#9ddfc7",
    "mma_issue": "#3564b7",
    "mma_wait": "#b9c9f2",
    "tmem_drain": "#7a4db3",
}

MARKERS = {
    "tma_done": "#087f5b",
    "mma_done": "#b42318",
    "mainloop_done": "#111827",
    "trace_done": "#7a4db3",
}


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("stage"):
                continue
            start = int(row["start"])
            end = int(row["end"])
            if end <= start:
                continue
            rows.append(
                {
                    "stage": row["stage"],
                    "iter": int(row["iter"]),
                    "warp": int(row["warp"]),
                    "start": start,
                    "end": end,
                    "cycles": int(row["cycles"]),
                }
            )
    return rows


def stage_slot(iteration):
    return iteration % 3


def pipe_for(row):
    if row["stage"] == "tma_issue":
        return row["warp"]
    if row["stage"] in ("tma_wait", "mma_issue", "mma_wait"):
        return row["warp"] - 2
    return row["warp"]


def lane_key(row):
    return (row["stage"], row["warp"])


def lane_name(key):
    for stage, warp, label in LANES:
        if key == (stage, warp):
            return label
    return f"{key[0]} w{key[1]}"


def short_label(row):
    kt = row["iter"]
    s = stage_slot(kt)
    pipe = pipe_for(row)
    if row["stage"] == "tma_issue":
        return f"TMA kt{kt} s{s} A+B0" if pipe == 0 else f"TMA kt{kt} s{s} B1"
    if row["stage"] == "tma_wait":
        return f"wait p{pipe} kt{kt}"
    if row["stage"] == "mma_issue":
        return f"MMA p{pipe} kt{kt}"
    if row["stage"] == "mma_wait":
        return f"wait MMA p{pipe} kt{kt}"
    if row["stage"] == "tmem_drain":
        tile = ("C00", "C01", "C10", "C11")[row["warp"]]
        return f"drain {tile}"
    return row["stage"]


def long_label(row):
    kt = row["iter"]
    s = stage_slot(kt)
    pipe = pipe_for(row)
    if row["stage"] == "tma_issue":
        if pipe == 0:
            return f"kt{kt} stage{s}: issue TMA A[256x64] + B0[64x128]"
        return f"kt{kt} stage{s}: issue TMA B1[64x128]"
    if row["stage"] == "tma_wait":
        return f"kt{kt} stage{s}: pipe{pipe} waits until A and B{pipe} are visible in shared memory"
    if row["stage"] == "mma_issue":
        return (
            f"kt{kt} stage{s}: pipe{pipe} issues "
            f"{'C00/C10' if pipe == 0 else 'C01/C11'}, "
            "two logical 128x128x64 MMA groups"
        )
    if row["stage"] == "mma_wait":
        return f"kt{kt} stage{s}: wait until pipe{pipe} tcgen05 MMA group completes"
    if row["stage"] == "tmem_drain":
        tile = ("C00", "C01", "C10", "C11")[row["warp"]]
        return f"final TMEM drain for {tile}"
    return short_label(row)


def marker_text(kind, row):
    kt = row["iter"]
    if kind == "tma_done":
        return f"p{pipe_for(row)} TMA done kt{kt}"
    if kind == "mma_done":
        return f"p{pipe_for(row)} MMA done kt{kt}"
    return f"done kt{kt}"


def ceil_to(value, step):
    return ((value + step - 1) // step) * step


def write_svg(path, rows, title):
    if not rows:
        raise SystemExit("trace CSV has no drawable rows")

    lane_order = [(stage, warp) for stage, warp, _ in LANES]
    seen = set(lane_order)
    for row in rows:
        key = lane_key(row)
        if key not in seen:
            lane_order.append(key)
            seen.add(key)
    lane_index = {key: idx for idx, key in enumerate(lane_order)}

    non_drain = [r for r in rows if r["stage"] != "tmem_drain"]
    if not non_drain:
        raise SystemExit("trace CSV has no non-drain rows")
    mainloop_done = max((r["end"] for r in rows if r["stage"] == "mma_wait"), default=max(r["end"] for r in non_drain))
    trace_done = max(r["end"] for r in rows)
    base = min(r["start"] for r in non_drain)
    axis_end = ceil_to(trace_done - base, 1000)
    scale = 0.095
    left = 245
    label_left = 22
    top = 86
    row_h = 58
    bar_h = 34
    right = 260
    width = int(left + axis_end * scale + right)
    height = top + len(lane_order) * row_h + 64

    def x(cycle):
        return left + (cycle - base) * scale

    def y_for(key):
        return top + lane_index[key] * row_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>',
        'text{font-family:Arial,sans-serif;fill:#1f2937}',
        '.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}',
        '.title{font-size:17px;font-weight:700}',
        '.subtitle{font-size:12px;fill:#4b5563}',
        '.label{font-size:13px;fill:#333}',
        '.barText{font-size:12px;fill:white;font-weight:700}',
        '.small{font-size:10px}',
        '.grid{stroke:#e3e8ef;stroke-width:1}',
        '.lane{stroke:#ccd5e0;stroke-width:1}',
        '</style>',
        f'<text class="title" x="{label_left}" y="26">{html.escape(title)}</text>',
        (
            f'<text class="subtitle mono" x="{label_left}" y="47">'
            f'base raw cycle={base}, mainloop done={mainloop_done - base} cyc, '
            f'trace done={trace_done - base} cyc</text>'
        ),
        (
            f'<text class="subtitle" x="{label_left}" y="66">'
            'TMA done markers use TMA wait end; MMA done markers use MMA wait end. '
            'pipe0 covers C00/C10 and pipe1 covers C01/C11.</text>'
        ),
    ]

    grid_step = 1000
    for grid in range(0, axis_end + 1, grid_step):
        gx = left + grid * scale
        parts.append(
            f'<line class="grid" x1="{gx:.1f}" y1="72" x2="{gx:.1f}" y2="{height - 36}"/>'
        )
        parts.append(
            f'<text class="mono small" x="{gx:.1f}" y="82" text-anchor="middle" fill="#667">{grid}</text>'
        )

    for idx, key in enumerate(lane_order):
        y = top + idx * row_h
        parts.append(
            f'<text class="label" x="{label_left}" y="{y + 20}">{html.escape(lane_name(key))}</text>'
        )
        parts.append(
            f'<line class="lane" x1="{left}" y1="{y + bar_h + 8}" '
            f'x2="{left + axis_end * scale:.1f}" y2="{y + bar_h + 8}"/>'
        )

    blocks = []
    for row in rows:
        start = row["start"]
        if row["stage"] == "tmem_drain":
            start = max(start, mainloop_done)
        end = max(row["end"], start + 1)
        blocks.append((start, end, row))

    for start, end, row in sorted(blocks, key=lambda item: (item[0], lane_index[lane_key(item[2])])):
        key = lane_key(row)
        y = y_for(key)
        bx = x(start)
        bw = max(1.0, (end - start) * scale)
        color = COLORS.get(row["stage"], "#5f6368")
        raw_note = ""
        if row["stage"] == "tmem_drain" and row["start"] < start:
            raw_note = f" raw_start={row['start'] - base} clamped_start={start - base}"
        parts.append(
            f'<rect x="{bx:.1f}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
            f'fill="{color}" stroke="#0f3768" stroke-width="1">'
            f'<title>{html.escape(long_label(row))}: '
            f'{end - start} cyc{raw_note}</title></rect>'
        )
        cx = bx + bw / 2
        label = short_label(row)
        if bw > 116:
            parts.append(
                f'<text class="barText" x="{cx:.1f}" y="{y + 15}" text-anchor="middle">'
                f'{html.escape(label)}</text>'
            )
            parts.append(
                f'<text class="barText small" x="{cx:.1f}" y="{y + 29}" text-anchor="middle">'
                f'{end - start} cyc</text>'
            )
        elif bw > 52:
            parts.append(
                f'<text class="barText small" x="{cx:.1f}" y="{y + 21}" text-anchor="middle">'
                f'{html.escape(label)}</text>'
            )

    def vertical_marker(cycle, y0, y1, color, text, text_y, dashed=False):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        mx = x(cycle)
        parts.append(
            f'<line x1="{mx:.1f}" y1="{y0:.1f}" x2="{mx:.1f}" y2="{y1:.1f}" '
            f'stroke="{color}" stroke-width="3"{dash}/>'
        )
        parts.append(
            f'<path d="M {mx - 5:.1f} {y0:.1f} L {mx + 5:.1f} {y0:.1f} L {mx:.1f} {y0 + 8:.1f} Z" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text class="mono small" x="{mx + 5:.1f}" y="{text_y:.1f}" fill="{color}">'
            f'{html.escape(text)}</text>'
        )

    for idx, row in enumerate(sorted((r for r in rows if r["stage"] == "tma_wait"), key=lambda r: r["iter"])):
        pipe = pipe_for(row)
        tma_top = y_for(("tma_issue", pipe)) - 4
        tma_bottom = y_for(("tma_wait", row["warp"])) + bar_h + 6
        vertical_marker(
            row["end"],
            tma_top,
            tma_bottom,
            MARKERS["tma_done"],
            marker_text("tma_done", row),
            tma_bottom + 12 + (idx & 1) * 10,
        )

    for idx, row in enumerate(sorted((r for r in rows if r["stage"] == "mma_wait"), key=lambda r: r["iter"])):
        mma_top = y_for(("mma_issue", row["warp"])) - 4
        mma_bottom = y_for(("mma_wait", row["warp"])) + bar_h + 6
        vertical_marker(
            row["end"],
            mma_top,
            mma_bottom,
            MARKERS["mma_done"],
            marker_text("mma_done", row),
            mma_bottom + 12 + (idx & 1) * 10,
        )

    vertical_marker(
        mainloop_done,
        top - 14,
        top + len(lane_order) * row_h - 12,
        MARKERS["mainloop_done"],
        f"MAINLOOP DONE / last MMA done ({mainloop_done - base} cyc)",
        top - 20,
        dashed=True,
    )
    vertical_marker(
        trace_done,
        top - 4,
        top + len(lane_order) * row_h - 4,
        MARKERS["trace_done"],
        f"TRACE END / last drain done ({trace_done - base} cyc)",
        height - 20,
        dashed=True,
    )

    legend_y = height - 38
    legend_x = left
    legend = [
        ("TMA issue", COLORS["tma_issue"]),
        ("TMA done", MARKERS["tma_done"]),
        ("MMA issue", COLORS["mma_issue"]),
        ("MMA done", MARKERS["mma_done"]),
        ("TMEM drain", COLORS["tmem_drain"]),
    ]
    for label, color in legend:
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text class="small" x="{legend_x + 18}" y="{legend_y + 11}">{label}</text>')
        legend_x += 112

    parts.append("</svg>\n")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--svg", required=True)
    p.add_argument("--title", default="5.MMA GEMM pipeline trace")
    args = p.parse_args()
    write_svg(args.svg, read_rows(args.trace), args.title)
    print(f"svg={args.svg}")


if __name__ == "__main__":
    main()
