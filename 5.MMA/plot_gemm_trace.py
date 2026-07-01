#!/usr/bin/env python3
import argparse
import csv
import html


STAGE_ROWS = {
    "tma_issue": "TMA issue",
    "tma_wait": "TMA wait",
    "mma_issue": "MMA issue",
    "mma_wait": "MMA wait",
    "tmem_drain": "TMEM drain",
}

COLORS = {
    "tma_issue": "#2f80ed",
    "tma_wait": "#9bbff5",
    "mma_issue": "#eb5757",
    "mma_wait": "#f4a5a5",
    "tmem_drain": "#27ae60",
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


def lane_name(row):
    if row["stage"] == "tmem_drain":
        return f"{STAGE_ROWS[row['stage']]} w{row['warp']}"
    return STAGE_ROWS.get(row["stage"], row["stage"])


def write_svg(path, rows, title):
    if not rows:
        raise SystemExit("trace CSV has no drawable rows")

    lanes = []
    seen = set()
    for stage in ("tma_issue", "tma_wait", "mma_issue", "mma_wait"):
        name = STAGE_ROWS[stage]
        lanes.append(name)
        seen.add(name)
    for warp in range(4):
        name = f"TMEM drain w{warp}"
        lanes.append(name)
        seen.add(name)
    for row in rows:
        name = lane_name(row)
        if name not in seen:
            lanes.append(name)
            seen.add(name)

    lane_to_y = {name: i for i, name in enumerate(lanes)}
    min_cycle = min(r["start"] for r in rows)
    max_cycle = max(r["end"] for r in rows)
    span = max(1, max_cycle - min_cycle)

    left = 150
    right = 32
    top = 56
    bottom = 48
    lane_h = 30
    plot_w = 1200
    width = left + plot_w + right
    height = top + bottom + lane_h * len(lanes)

    def x(cycle):
        return left + (cycle - min_cycle) * plot_w / span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;fill:#202124}",
        ".title{font-size:16px;font-weight:700}",
        ".axis{stroke:#9aa0a6;stroke-width:1}",
        ".grid{stroke:#e0e3e7;stroke-width:1}",
        ".label{fill:#3c4043}",
        ".bar{rx:3;ry:3}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="24" y="28">{html.escape(title)}</text>',
        f'<text x="24" y="46">window: {min_cycle}..{max_cycle} cycles, span={span}</text>',
    ]

    tick_count = 8
    for i in range(tick_count + 1):
        cycle = min_cycle + span * i // tick_count
        tx = x(cycle)
        parts.append(f'<line class="grid" x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}"/>')
        parts.append(f'<text x="{tx - 18:.1f}" y="{height - 18}">{cycle - min_cycle}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{left+plot_w}" y2="{height-bottom}"/>')
    parts.append(f'<text x="{left + plot_w/2 - 54:.1f}" y="{height - 4}">cycles from window start</text>')

    for lane, idx in lane_to_y.items():
        y = top + idx * lane_h
        parts.append(f'<text class="label" x="18" y="{y + 19}">{html.escape(lane)}</text>')
        parts.append(f'<line class="grid" x1="{left}" y1="{y + lane_h}" x2="{left + plot_w}" y2="{y + lane_h}"/>')

    for row in rows:
        y = top + lane_to_y[lane_name(row)] * lane_h + 6
        x0 = x(row["start"])
        x1 = x(row["end"])
        w = max(1.0, x1 - x0)
        color = COLORS.get(row["stage"], "#5f6368")
        parts.append(
            f'<rect class="bar" x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" height="18" '
            f'fill="{color}"><title>iter={row["iter"]} stage={row["stage"]} '
            f'warp={row["warp"]} cycles={row["cycles"]}</title></rect>'
        )
        if row["stage"] in ("tma_issue", "mma_issue") and w > 18:
            parts.append(
                f'<text x="{x0 + 3:.1f}" y="{y + 13:.1f}" fill="#fff">{row["iter"]}</text>'
            )

    legend_x = left + 8
    legend_y = top - 22
    for stage in ("tma_issue", "tma_wait", "mma_issue", "mma_wait", "tmem_drain"):
        color = COLORS[stage]
        label = STAGE_ROWS[stage]
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{legend_y + 11}">{label}</text>')
        legend_x += 116

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
