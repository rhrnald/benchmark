#!/usr/bin/env python3
"""Render the current E7a GEMM clock64 phase trace as a standalone SVG.

The E7a trace CSV contains two kinds of records:

* interval: a real begin/end interval on one warp;
* aggregate: the envelope and the sum of many individually timed calls.

Only interval records are placed on the timeline.  Aggregate records are shown
as independent bars in the lower panel because their envelopes overlap and
must not be interpreted as an additive critical path.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path


OVERVIEW_EVENTS = {
    "scheduler_atomic": ("scheduler", "#94a3b8"),
    "scheduler_barrier": ("scheduler", "#cbd5e1"),
    "scheduler_decode": ("scheduler", "#64748b"),
    "producer_reuse_wait_and_tma_issue_loop": ("producer mainloop", "#38bdf8"),
    "consumer_mainloop_and_final_drain": ("consumer mainloop", "#a78bfa"),
    "consumer_final_mma_drain": ("final MMA drain", "#f97316"),
    "mainloop_join_barrier": ("mainloop join", "#475569"),
    "epilogue_chunk_0_stage_and_tma_issue": ("C chunk 0", "#22c55e"),
    "epilogue_chunk_1_stage_and_tma_issue": ("C chunk 1", "#16a34a"),
    "epilogue_chunk_2_stage_and_tma_issue": ("C chunk 2", "#14b8a6"),
    "epilogue_chunk_3_stage_and_tma_issue": ("C chunk 3", "#0d9488"),
    "epilogue_group_0_commit_wait_and_barrier": ("C group 0 drain", "#f59e0b"),
    "epilogue_group_1_commit_wait_and_barrier": ("C group 1 drain", "#d97706"),
    "final_tile_barrier": ("tile barrier", "#334155"),
}

BREAKDOWN_EVENTS = [
    ("consumer_wait_a", "wait A ready", "#0ea5e9"),
    ("consumer_wait_b0", "wait B0 ready", "#06b6d4"),
    ("consumer_mma_b0_issue", "issue MMA B0 (2/4)", "#8b5cf6"),
    ("consumer_wait_b1", "wait B1 ready", "#14b8a6"),
    ("consumer_mma_b1_issue", "issue MMA B1 (2/4)", "#a855f7"),
    ("consumer_mma_commit", "MMA commit", "#ec4899"),
    ("consumer_final_mma_drain", "final MMA drain", "#f97316"),
]


def read_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, object] = dict(raw)
            for key in (
                "size",
                "sm_id",
                "block_idx",
                "tile_iter",
                "linear_tile",
                "tile_m",
                "tile_n",
                "slot",
                "warp",
                "start_rel",
                "end_rel",
                "envelope_cycles",
                "total_cycles",
                "sample_count",
            ):
                row[key] = int(raw[key])
            row["avg_cycles_per_sample"] = float(raw["avg_cycles_per_sample"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty trace")
    return rows


def one(rows: list[dict[str, object]], event: str, warp: int) -> dict[str, object]:
    matches = [
        row for row in rows if row["event"] == event and row["warp"] == warp
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {event} record for warp {warp}, got {len(matches)}"
        )
    return matches[0]


def tile_end(rows: list[dict[str, object]]) -> int:
    final = [int(row["end_rel"]) for row in rows if row["event"] == "final_tile_barrier"]
    if not final:
        raise ValueError("trace has no final_tile_barrier records")
    return max(final)


def representative_run(
    runs: list[tuple[Path, list[dict[str, object]]]]
) -> tuple[Path, list[dict[str, object]]]:
    ends = [tile_end(rows) for _, rows in runs]
    target = statistics.median(ends)
    return min(runs, key=lambda item: abs(tile_end(item[1]) - target))


def median_range(
    runs: list[tuple[Path, list[dict[str, object]]]], event: str, warp: int
) -> tuple[float, int, int, float, int]:
    records = [one(rows, event, warp) for _, rows in runs]
    totals = [int(row["total_cycles"]) for row in records]
    averages = [float(row["avg_cycles_per_sample"]) for row in records]
    samples = [int(row["sample_count"]) for row in records]
    return (
        statistics.median(totals),
        min(totals),
        max(totals),
        statistics.median(averages),
        int(statistics.median(samples)),
    )


def nice_step(span: float, target_ticks: int = 9) -> int:
    raw = max(span / target_ticks, 1.0)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return int(nice * magnitude)


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    fill: str = "#0f172a",
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">'
        f"{html.escape(value)}</text>"
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    title: str,
    *,
    opacity: float = 1.0,
    stroke: str = "none",
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 1.0):.2f}" '
        f'height="{height:.2f}" rx="3" fill="{fill}" opacity="{opacity:.3f}" '
        f'stroke="{stroke}"><title>{html.escape(title)}</title></rect>'
    )


def build_svg(
    runs: list[tuple[Path, list[dict[str, object]]]],
    title: str,
) -> tuple[str, list[dict[str, object]]]:
    rep_path, rows = representative_run(runs)
    meta = rows[0]
    width = 1680
    height = 1300
    left = 188
    right = 55
    plot_width = width - left - right
    overview_top = 175
    lane_height = 54
    bar_height = 24
    overview_bottom = overview_top + 4 * lane_height
    end_cycle = tile_end(rows)
    x_scale = plot_width / max(end_cycle, 1)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">'
        '<feDropShadow dx="0" dy="1" stdDeviation="1.5" '
        'flood-color="#0f172a" flood-opacity="0.12"/></filter>',
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        svg_text(45, 48, title, size=27, weight=700),
        svg_text(
            45,
            78,
            (
                f'16K BF16 GEMM · E7a dual-wide · input={meta["input"]} · '
                f'{len(runs)} independent trace{"s" if len(runs) != 1 else ""}'
            ),
            size=15,
            fill="#475569",
        ),
        svg_text(
            45,
            102,
            (
                f'representative={rep_path.name} · block={meta["block_idx"]} '
                f'SM={meta["sm_id"]} · persistent tile_iter={meta["tile_iter"]} '
                f'· output tile=({meta["tile_m"]},{meta["tile_n"]})'
            ),
            size=13,
            fill="#64748b",
        ),
        svg_text(45, 143, "A. One output-tile timeline", size=18, weight=700),
        svg_text(
            width - 55,
            143,
            "device clock64 cycles (same CTA / same SM)",
            size=12,
            fill="#64748b",
            anchor="end",
        ),
    ]

    tick_step = nice_step(end_cycle)
    for cycle in range(0, end_cycle + tick_step, tick_step):
        if cycle > end_cycle:
            break
        x = left + cycle * x_scale
        out.append(
            f'<line x1="{x:.2f}" y1="{overview_top - 18}" '
            f'x2="{x:.2f}" y2="{overview_bottom + 12}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        out.append(
            svg_text(x, overview_top - 24, f"{cycle:,}", size=11,
                     fill="#64748b", anchor="middle")
        )

    lane_labels = {
        0: "W0 · A + late B1 TMA",
        1: "W1 · early B0 TMA",
        2: "W2 · M0 MMA consumer",
        3: "W3 · M1 MMA consumer",
    }
    for warp in range(4):
        y = overview_top + warp * lane_height
        out.append(
            f'<rect x="35" y="{y - 9:.2f}" width="{width - 70}" '
            f'height="{lane_height - 5}" rx="7" '
            f'fill="{"#ffffff" if warp % 2 == 0 else "#f1f5f9"}"/>'
        )
        out.append(svg_text(left - 14, y + 15, lane_labels[warp],
                            size=13, weight=600, anchor="end"))

    interval_rows = [
        row
        for row in rows
        if row["kind"] == "interval" and row["event"] in OVERVIEW_EVENTS
    ]
    draw_priority = {
        "scheduler_atomic": 0,
        "scheduler_barrier": 1,
        "scheduler_decode": 2,
        "producer_reuse_wait_and_tma_issue_loop": 0,
        "consumer_mainloop_and_final_drain": 0,
        "mainloop_join_barrier": 2,
        "epilogue_chunk_0_stage_and_tma_issue": 0,
        "epilogue_chunk_1_stage_and_tma_issue": 0,
        "epilogue_chunk_2_stage_and_tma_issue": 0,
        "epilogue_chunk_3_stage_and_tma_issue": 0,
        "epilogue_group_0_commit_wait_and_barrier": 2,
        "epilogue_group_1_commit_wait_and_barrier": 2,
        "consumer_final_mma_drain": 3,
        "final_tile_barrier": 3,
    }
    interval_rows.sort(key=lambda row: draw_priority.get(str(row["event"]), 0))
    for row in interval_rows:
        event = str(row["event"])
        label, color = OVERVIEW_EVENTS[event]
        warp = int(row["warp"])
        x0 = left + int(row["start_rel"]) * x_scale
        x1 = left + int(row["end_rel"]) * x_scale
        y = overview_top + warp * lane_height
        duration = int(row["end_rel"]) - int(row["start_rel"])
        tooltip = (
            f"{label} | W{warp} | {int(row['start_rel']):,}–"
            f"{int(row['end_rel']):,} cycles | duration={duration:,}"
        )
        if event in {"consumer_final_mma_drain", "final_tile_barrier"}:
            out.append(
                svg_rect(x0, y + 1, x1 - x0, bar_height - 2, color, tooltip,
                         opacity=0.95, stroke="#ffffff")
            )
        elif "group_" in event or event in {
            "mainloop_join_barrier",
            "scheduler_barrier",
            "scheduler_decode",
        }:
            out.append(
                svg_rect(x0, y + 3, x1 - x0, bar_height - 6, color, tooltip,
                         opacity=0.96)
            )
        else:
            out.append(svg_rect(x0, y, x1 - x0, bar_height, color, tooltip,
                                opacity=0.88))

    overview_legend = [
        ("scheduler", "#64748b"),
        ("TMA producer loop", "#38bdf8"),
        ("MMA consumer loop", "#a78bfa"),
        ("C stage/TMA store", "#22c55e"),
        ("store-group drain", "#f59e0b"),
    ]
    legend_y = overview_bottom + 47
    lx = left
    for label, color in overview_legend:
        out.append(
            f'<rect x="{lx}" y="{legend_y - 12}" width="14" height="14" '
            f'rx="2" fill="{color}"/>'
        )
        out.append(svg_text(lx + 20, legend_y, label, size=12, fill="#475569"))
        lx += 42 + len(label) * 7.1

    scheduler_start = min(
        int(row["start_rel"])
        for row in rows
        if str(row["event"]).startswith("scheduler_")
    )
    scheduler_end = max(
        int(row["end_rel"])
        for row in rows
        if str(row["event"]).startswith("scheduler_")
    )
    main_start = min(
        int(row["start_rel"])
        for row in rows
        if row["event"]
        in {
            "producer_reuse_wait_and_tma_issue_loop",
            "consumer_mainloop_and_final_drain",
        }
    )
    main_end = max(
        int(row["end_rel"])
        for row in rows
        if row["event"] == "mainloop_join_barrier"
    )
    epi_start = min(
        int(row["start_rel"]) for row in rows if row["event"] == "epilogue_complete"
    )
    epi_end = max(
        int(row["end_rel"]) for row in rows if row["event"] == "epilogue_complete"
    )
    producer_end = [
        int(one(rows, "producer_reuse_wait_and_tma_issue_loop", warp)["end_rel"])
        for warp in (0, 1)
    ]
    consumer_end = [
        int(one(rows, "consumer_mainloop_and_final_drain", warp)["end_rel"])
        for warp in (2, 3)
    ]

    card_y = overview_bottom + 82
    card_width = (width - 90 - 3 * 16) / 4
    cards = [
        ("scheduler", scheduler_end - scheduler_start,
         100.0 * (scheduler_end - scheduler_start) / end_cycle),
        ("mainloop → CTA join", main_end - main_start,
         100.0 * (main_end - main_start) / end_cycle),
        ("FP32 C epilogue", epi_end - epi_start,
         100.0 * (epi_end - epi_start) / end_cycle),
        ("complete output tile", end_cycle, 100.0),
    ]
    for index, (label, cycles, percent) in enumerate(cards):
        x = 45 + index * (card_width + 16)
        out.append(
            f'<rect x="{x:.2f}" y="{card_y:.2f}" width="{card_width:.2f}" '
            'height="78" rx="10" fill="#ffffff" stroke="#e2e8f0" '
            'filter="url(#shadow)"/>'
        )
        out.append(svg_text(x + 16, card_y + 25, label, size=12, fill="#64748b"))
        out.append(
            svg_text(x + 16, card_y + 53, f"{cycles:,} cyc",
                     size=20, weight=700)
        )
        out.append(
            svg_text(x + card_width - 14, card_y + 53, f"{percent:.2f}%",
                     size=12, fill="#64748b", anchor="end")
        )

    breakdown_top = card_y + 128
    out.extend(
        [
            svg_text(45, breakdown_top, "B. Consumer call-time aggregates",
                     size=18, weight=700),
            svg_text(
                45,
                breakdown_top + 25,
                (
                    "Median total across runs; whisker = min–max. "
                    "Rows overlap in time and are not an additive critical path."
                ),
                size=12,
                fill="#64748b",
            ),
        ]
    )
    chart_top = breakdown_top + 80
    chart_left = 300
    chart_right = width - 70
    chart_width = chart_right - chart_left
    row_height = 58

    all_medians = []
    summary_rows: list[dict[str, object]] = []
    for event, label, _ in BREAKDOWN_EVENTS:
        for warp in (2, 3):
            median, minimum, maximum, avg_sample, samples = median_range(
                runs, event, warp
            )
            all_medians.append(maximum)
            summary_rows.append(
                {
                    "event": event,
                    "label": label,
                    "warp": warp,
                    "runs": len(runs),
                    "total_median": median,
                    "total_min": minimum,
                    "total_max": maximum,
                    "avg_sample_median": avg_sample,
                    "sample_count": samples,
                }
            )
    aggregate_max = max(all_medians)
    aggregate_step = nice_step(aggregate_max, target_ticks=6)
    aggregate_limit = int(math.ceil(aggregate_max / aggregate_step) * aggregate_step)
    aggregate_scale = chart_width / max(aggregate_limit, 1)

    for cycle in range(0, aggregate_limit + 1, aggregate_step):
        x = chart_left + cycle * aggregate_scale
        out.append(
            f'<line x1="{x:.2f}" y1="{chart_top - 20}" '
            f'x2="{x:.2f}" y2="{chart_top + len(BREAKDOWN_EVENTS) * row_height}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        out.append(
            svg_text(x, chart_top - 27, f"{cycle:,}", size=11,
                     fill="#64748b", anchor="middle")
        )

    for row_index, (event, label, color) in enumerate(BREAKDOWN_EVENTS):
        y = chart_top + row_index * row_height
        out.append(svg_text(chart_left - 22, y + 25, label, size=13,
                            weight=600, anchor="end"))
        for warp_offset, warp in enumerate((2, 3)):
            median, minimum, maximum, avg_sample, samples = median_range(
                runs, event, warp
            )
            bar_y = y + 5 + warp_offset * 20
            whisker_y = bar_y + 7
            min_x = chart_left + minimum * aggregate_scale
            max_x = chart_left + maximum * aggregate_scale
            median_x = chart_left + median * aggregate_scale
            out.append(
                f'<line x1="{min_x:.2f}" y1="{whisker_y:.2f}" '
                f'x2="{max_x:.2f}" y2="{whisker_y:.2f}" '
                f'stroke="{color}" stroke-width="2" opacity="0.65"/>'
            )
            out.append(
                f'<line x1="{min_x:.2f}" y1="{whisker_y - 4:.2f}" '
                f'x2="{min_x:.2f}" y2="{whisker_y + 4:.2f}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            out.append(
                f'<line x1="{max_x:.2f}" y1="{whisker_y - 4:.2f}" '
                f'x2="{max_x:.2f}" y2="{whisker_y + 4:.2f}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            tooltip = (
                f"{label} | W{warp} | median total={median:g} cycles | "
                f"range={minimum:,}–{maximum:,} | samples={samples} | "
                f"median/sample={avg_sample:.2f}"
            )
            out.append(
                svg_rect(chart_left, bar_y, median_x - chart_left, 14, color,
                         tooltip, opacity=0.82)
            )
            out.append(
                f'<circle cx="{median_x:.2f}" cy="{whisker_y:.2f}" r="3.5" '
                f'fill="{color}"><title>{html.escape(tooltip)}</title></circle>'
            )
            out.append(
                svg_text(
                    max(median_x + 8, chart_left + 8),
                    bar_y + 12,
                    (
                        f"W{warp} {median:g}"
                        + (f" ({avg_sample:.1f}/call)" if samples > 1 else "")
                    ),
                    size=10,
                    fill="#334155",
                )
            )

    notes_y = chart_top + len(BREAKDOWN_EVENTS) * row_height + 35
    producer_lead = max(consumer_end) - max(producer_end)
    consumer_skew = abs(consumer_end[0] - consumer_end[1])
    out.extend(
        [
            f'<rect x="45" y="{notes_y:.2f}" width="{width - 90}" height="88" '
            'rx="10" fill="#fff7ed" stroke="#fed7aa"/>',
            svg_text(65, notes_y + 27, "Reading this trace", size=13,
                     weight=700, fill="#9a3412"),
            svg_text(
                65,
                notes_y + 51,
                (
                    f"In the representative run, the last producer returns "
                    f"{producer_lead:,} cycles before the last consumer; "
                    f"W2/W3 consumer finish skew is {consumer_skew:,} cycles."
                ),
                size=12,
                fill="#7c2d12",
            ),
            svg_text(
                65,
                notes_y + 72,
                (
                    "TMA bars measure issue-loop occupancy, not transfer completion. "
                    "MMA bars measure descriptor setup/issue, not isolated tensor-core execution."
                ),
                size=12,
                fill="#7c2d12",
            ),
            svg_text(
                width - 45,
                height - 20,
                (
                    "Diagnostic trace only: clock reads and trace storage perturb "
                    "the kernel; use the uninstrumented binary for TFLOP/s."
                ),
                size=11,
                fill="#64748b",
                anchor="end",
            ),
            "</svg>",
        ]
    )
    return "\n".join(out), summary_rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "event",
        "label",
        "warp",
        "runs",
        "total_median",
        "total_min",
        "total_max",
        "avg_sample_median",
        "sample_count",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_csv", nargs="+", type=Path)
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument(
        "--title", default="B200 GEMM clock64 trace — current E7a kernel"
    )
    args = parser.parse_args()

    runs = [(path, read_trace(path)) for path in args.trace_csv]
    sizes = {int(rows[0]["size"]) for _, rows in runs}
    inputs = {str(rows[0]["input"]) for _, rows in runs}
    if len(sizes) != 1 or len(inputs) != 1:
        raise SystemExit(
            f"all traces must share size and input; sizes={sizes}, inputs={inputs}"
        )

    svg, summary = build_svg(runs, args.title)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(svg)
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary_csv, summary)

    print(f"svg={args.svg} traces={len(runs)} size={next(iter(sizes))} input={next(iter(inputs))}")
    if args.summary_csv:
        print(f"summary_csv={args.summary_csv}")


if __name__ == "__main__":
    main()
