#!/usr/bin/env python3
"""Render an attention-style per-K-stage E7a GEMM pipeline trace.

The trace separates synchronous issue spans, explicit mbarrier wait spans, and
asynchronous completion observations.  "Observed" is deliberate: clock64 sees
when another warp passes a dependency, not the exact hardware completion edge.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from pathlib import Path


EVENT_STYLE = {
    "p0_wait_mma_m0": ("wait MMA M0 reuse", "#94a3b8"),
    "p0_wait_mma_m1": ("wait MMA M1 reuse", "#64748b"),
    "p0_prepare_and_issue_tma_a": ("prepare + issue A TMA", "#22c55e"),
    "p0_prepare_and_issue_tma_b1": ("prepare + issue B1 TMA", "#14b8a6"),
    "p1_wait_mma_m0": ("wait MMA M0 reuse", "#94a3b8"),
    "p1_wait_mma_m1": ("wait MMA M1 reuse", "#64748b"),
    "p1_prepare_and_issue_tma_b0": ("prepare + issue B0 TMA", "#06b6d4"),
    "c2_wait_tma_a": ("wait A ready", "#fb7185"),
    "c2_wait_tma_b0": ("wait B0 ready", "#f97316"),
    "c2_prepare_and_issue_mma_b0": ("prepare + issue MMA B0", "#3b82f6"),
    "c2_wait_tma_b1": ("wait B1 ready", "#f59e0b"),
    "c2_prepare_and_issue_mma_b1": ("prepare + issue MMA B1", "#8b5cf6"),
    "c2_commit_mma": ("commit MMA", "#ec4899"),
    "c3_wait_tma_a": ("wait A ready", "#fb7185"),
    "c3_wait_tma_b0": ("wait B0 ready", "#f97316"),
    "c3_prepare_and_issue_mma_b0": ("prepare + issue MMA B0", "#3b82f6"),
    "c3_wait_tma_b1": ("wait B1 ready", "#f59e0b"),
    "c3_prepare_and_issue_mma_b1": ("prepare + issue MMA B1", "#8b5cf6"),
    "c3_commit_mma": ("commit MMA", "#ec4899"),
}

WARP_LABELS = {
    0: "W0 · A / late-B1 producer",
    1: "W1 · early-B0 producer",
    2: "W2 · M0 MMA consumer",
    3: "W3 · M1 MMA consumer",
}

SLOT_COLORS = {0: "#ef4444", 1: "#f59e0b", 2: "#2563eb"}

CORE_EVENTS = {
    0: [
        "p0_wait_mma_m0",
        "p0_wait_mma_m1",
        "p0_prepare_and_issue_tma_a",
        "p0_prepare_and_issue_tma_b1",
    ],
    1: [
        "p1_wait_mma_m0",
        "p1_wait_mma_m1",
        "p1_prepare_and_issue_tma_b0",
    ],
    2: [
        "c2_wait_tma_a",
        "c2_wait_tma_b0",
        "c2_prepare_and_issue_mma_b0",
        "c2_wait_tma_b1",
        "c2_prepare_and_issue_mma_b1",
        "c2_commit_mma",
    ],
    3: [
        "c3_wait_tma_a",
        "c3_wait_tma_b0",
        "c3_prepare_and_issue_mma_b0",
        "c3_wait_tma_b1",
        "c3_prepare_and_issue_mma_b1",
        "c3_commit_mma",
    ],
}


def read_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    integer_fields = {
        "size",
        "sm_id",
        "block_idx",
        "tile_iter",
        "linear_tile",
        "tile_m",
        "tile_n",
        "ktiles",
        "k_start",
        "k_count",
        "core_k_count",
        "base_clock",
        "kt",
        "stage_epoch",
        "ring_stage",
        "phase",
        "event_id",
        "related_kt",
        "warp",
        "start_raw",
        "end_raw",
        "start_rel",
        "end_rel",
        "cycles",
    }
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, object] = dict(raw)
            for key in integer_fields:
                row[key] = int(raw[key])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty pipeline trace")
    metadata = {
        (
            row["size"],
            row["input"],
            row["sm_id"],
            row["block_idx"],
            row["tile_iter"],
            row["linear_tile"],
            row["tile_m"],
            row["tile_n"],
            row["ktiles"],
            row["k_start"],
            row["k_count"],
            row["core_k_count"],
            row["base_clock"],
        )
        for row in rows
    }
    if len(metadata) != 1:
        raise ValueError(f"{path}: inconsistent metadata")
    core_count = int(rows[0]["core_k_count"])
    context_count = int(rows[0]["k_count"]) - core_count
    expected = core_count * 19 + context_count * 4
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, got {len(rows)}")
    k_start = int(rows[0]["k_start"])
    core_end = k_start + core_count
    reuse_events = {
        "p0_wait_mma_m0",
        "p0_wait_mma_m1",
        "p1_wait_mma_m0",
        "p1_wait_mma_m1",
    }
    expected_warp = {
        name: warp for warp, names in CORE_EVENTS.items() for name in names
    }
    for row in rows:
        kt = int(row["kt"])
        name = str(row["event"])
        stage_epoch = int(row["tile_iter"]) * int(row["ktiles"]) + kt
        if int(row["stage_epoch"]) != stage_epoch:
            raise ValueError(f"{path}: bad stage_epoch for kt={kt} event={name}")
        if int(row["ring_stage"]) != stage_epoch % 3:
            raise ValueError(f"{path}: bad ring_stage for kt={kt} event={name}")
        if int(row["warp"]) != expected_warp[name]:
            raise ValueError(f"{path}: bad warp for kt={kt} event={name}")
        if kt >= core_end and name not in reuse_events:
            raise ValueError(f"{path}: non-reuse context event kt={kt} {name}")
        reuse = name in reuse_events
        expected_phase = (
            ((stage_epoch - 3) // 3) & 1 if reuse else (stage_epoch // 3) & 1
        )
        expected_related = kt - 3 if reuse else -1
        if int(row["phase"]) != expected_phase:
            raise ValueError(f"{path}: bad phase for kt={kt} event={name}")
        if int(row["related_kt"]) != expected_related:
            raise ValueError(f"{path}: bad related_kt for kt={kt} event={name}")
        if int(row["end_raw"]) <= int(row["start_raw"]):
            raise ValueError(f"{path}: non-positive interval kt={kt} event={name}")
        if int(row["start_raw"]) - int(row["base_clock"]) != int(row["start_rel"]):
            raise ValueError(f"{path}: bad start_rel for kt={kt} event={name}")
        if int(row["end_raw"]) - int(row["base_clock"]) != int(row["end_rel"]):
            raise ValueError(f"{path}: bad end_rel for kt={kt} event={name}")
        if int(row["end_raw"]) - int(row["start_raw"]) != int(row["cycles"]):
            raise ValueError(f"{path}: bad cycles for kt={kt} event={name}")
    return rows


def index_rows(rows: list[dict[str, object]]) -> dict[tuple[int, str], dict[str, object]]:
    index: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["kt"]), str(row["event"]))
        if key in index:
            raise ValueError(f"duplicate trace key {key}")
        index[key] = row
    return index


def event(
    index: dict[tuple[int, str], dict[str, object]], kt: int, name: str
) -> dict[str, object]:
    try:
        return index[(kt, name)]
    except KeyError as exc:
        raise ValueError(f"missing kt={kt} event={name}") from exc


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def nice_step(span: float, target: int = 10) -> int:
    raw = max(span / target, 1.0)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        value = 1
    elif normalized <= 2:
        value = 2
    elif normalized <= 5:
        value = 5
    else:
        value = 10
    return int(value * magnitude)


def text(
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


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    tooltip: str,
    *,
    opacity: float = 1.0,
    stroke: str = "none",
    dash: str = "",
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 1.5):.2f}" '
        f'height="{height:.2f}" rx="3" fill="{fill}" opacity="{opacity:.3f}" '
        f'stroke="{stroke}" stroke-width="1.5"{dash_attr}>'
        f"<title>{html.escape(tooltip)}</title></rect>"
    )


def tma_observation(
    idx: dict[tuple[int, str], dict[str, object]], kt: int, operand: str
) -> dict[str, float]:
    if operand == "A":
        issue_name = "p0_prepare_and_issue_tma_a"
        waits = ("c2_wait_tma_a", "c3_wait_tma_a")
    elif operand == "B0":
        issue_name = "p1_prepare_and_issue_tma_b0"
        waits = ("c2_wait_tma_b0", "c3_wait_tma_b0")
    elif operand == "B1":
        issue_name = "p0_prepare_and_issue_tma_b1"
        waits = ("c2_wait_tma_b1", "c3_wait_tma_b1")
    else:
        raise ValueError(operand)
    issue = event(idx, kt, issue_name)
    ready = [event(idx, kt, name) for name in waits]
    issue_end = int(issue["end_raw"])
    wait_starts = [int(row["start_raw"]) for row in ready]
    wait_ends = [int(row["end_raw"]) for row in ready]
    return {
        "issue_start": int(issue["start_raw"]),
        "issue_end": issue_end,
        "first_wait_start": min(wait_starts),
        "first_observed": min(wait_ends),
        "all_observed": max(wait_ends),
        "issue_to_first_pass_bound": min(wait_ends) - int(issue["start_raw"]),
        "post_issue_gap": min(wait_ends) - issue_end,
        "prefetch_lead": min(wait_starts) - issue_end,
    }


def mma_observation(
    idx: dict[tuple[int, str], dict[str, object]], kt: int, warp: int
) -> dict[str, float]:
    suffix = "2" if warp == 2 else "3"
    mblock = 0 if warp == 2 else 1
    issue0 = event(idx, kt, f"c{suffix}_prepare_and_issue_mma_b0")
    commit = event(idx, kt, f"c{suffix}_commit_mma")
    reuse_kt = kt + 3
    observations = [
        event(idx, reuse_kt, f"p0_wait_mma_m{mblock}"),
        event(idx, reuse_kt, f"p1_wait_mma_m{mblock}"),
    ]
    observed_ends = [int(row["end_raw"]) for row in observations]
    commit_end = int(commit["end_raw"])
    return {
        "issue_start": int(issue0["start_raw"]),
        "commit_start": int(commit["start_raw"]),
        "commit_end": commit_end,
        "first_observed": min(observed_ends),
        "all_observed": max(observed_ends),
        "prepare_to_first_reuse_pass_bound":
            min(observed_ends) - int(issue0["start_raw"]),
        "post_commit_gap": min(observed_ends) - commit_end,
        "reuse_kt": reuse_kt,
    }


def compute_metrics(
    rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    idx = index_rows(rows)
    k_start = int(rows[0]["k_start"])
    core_count = int(rows[0]["core_k_count"])
    core_kts = list(range(k_start, k_start + core_count))
    metric_rows: list[dict[str, object]] = []

    for position, kt in enumerate(core_kts):
        tma = {name: tma_observation(idx, kt, name) for name in ("A", "B0", "B1")}
        mma2 = mma_observation(idx, kt, 2)
        mma3 = mma_observation(idx, kt, 3)
        w2_waits = [
            int(event(idx, kt, name)["cycles"])
            for name in ("c2_wait_tma_a", "c2_wait_tma_b0", "c2_wait_tma_b1")
        ]
        w3_waits = [
            int(event(idx, kt, name)["cycles"])
            for name in ("c3_wait_tma_a", "c3_wait_tma_b0", "c3_wait_tma_b1")
        ]
        w0_reuse = [
            int(event(idx, kt, name)["cycles"])
            for name in ("p0_wait_mma_m0", "p0_wait_mma_m1")
        ]
        w1_reuse = [
            int(event(idx, kt, name)["cycles"])
            for name in ("p1_wait_mma_m0", "p1_wait_mma_m1")
        ]
        next_kt = kt + 1
        if position + 1 < len(core_kts):
            cadence2 = int(event(idx, next_kt, "c2_commit_mma")["end_raw"]) - int(
                event(idx, kt, "c2_commit_mma")["end_raw"]
            )
            cadence3 = int(event(idx, next_kt, "c3_commit_mma")["end_raw"]) - int(
                event(idx, kt, "c3_commit_mma")["end_raw"]
            )
        else:
            cadence2 = ""
            cadence3 = ""
        metric_rows.append(
            {
                "kt": kt,
                "ring_stage":
                    int(event(idx, kt, "p0_prepare_and_issue_tma_a")["ring_stage"]),
                "phase":
                    int(event(idx, kt, "p0_prepare_and_issue_tma_a")["phase"]),
                "w2_wait_a": w2_waits[0],
                "w2_wait_b0": w2_waits[1],
                "w2_wait_b1": w2_waits[2],
                "w2_wait_total": sum(w2_waits),
                "w3_wait_a": w3_waits[0],
                "w3_wait_b0": w3_waits[1],
                "w3_wait_b1": w3_waits[2],
                "w3_wait_total": sum(w3_waits),
                "w0_reuse_m0": w0_reuse[0],
                "w0_reuse_m1": w0_reuse[1],
                "w0_reuse_total": sum(w0_reuse),
                "w1_reuse_m0": w1_reuse[0],
                "w1_reuse_m1": w1_reuse[1],
                "w1_reuse_total": sum(w1_reuse),
                "a_prefetch_lead": int(tma["A"]["prefetch_lead"]),
                "a_issue_to_first_pass_bound":
                    int(tma["A"]["issue_to_first_pass_bound"]),
                "a_post_issue_gap": int(tma["A"]["post_issue_gap"]),
                "b0_prefetch_lead": int(tma["B0"]["prefetch_lead"]),
                "b0_issue_to_first_pass_bound":
                    int(tma["B0"]["issue_to_first_pass_bound"]),
                "b0_post_issue_gap": int(tma["B0"]["post_issue_gap"]),
                "b1_prefetch_lead": int(tma["B1"]["prefetch_lead"]),
                "b1_issue_to_first_pass_bound":
                    int(tma["B1"]["issue_to_first_pass_bound"]),
                "b1_post_issue_gap": int(tma["B1"]["post_issue_gap"]),
                "w2_prepare_to_first_reuse_pass_bound":
                    int(mma2["prepare_to_first_reuse_pass_bound"]),
                "w2_post_commit_gap": int(mma2["post_commit_gap"]),
                "w3_prepare_to_first_reuse_pass_bound":
                    int(mma3["prepare_to_first_reuse_pass_bound"]),
                "w3_post_commit_gap": int(mma3["post_commit_gap"]),
                "w2_commit_cadence": cadence2,
                "w3_commit_cadence": cadence3,
            }
        )

    commit_cadence2 = [
        float(row["w2_commit_cadence"])
        for row in metric_rows
        if row["w2_commit_cadence"] != ""
    ]
    commit_cadence3 = [
        float(row["w3_commit_cadence"])
        for row in metric_rows
        if row["w3_commit_cadence"] != ""
    ]
    wait_total2 = sum(float(row["w2_wait_total"]) for row in metric_rows)
    wait_total3 = sum(float(row["w3_wait_total"]) for row in metric_rows)
    span2 = int(event(idx, core_kts[-1], "c2_commit_mma")["end_raw"]) - int(
        event(idx, core_kts[0], "c2_wait_tma_a")["start_raw"]
    )
    span3 = int(event(idx, core_kts[-1], "c3_commit_mma")["end_raw"]) - int(
        event(idx, core_kts[0], "c3_wait_tma_a")["start_raw"]
    )
    producer_wait0 = sum(
        int(event(idx, kt, name)["cycles"])
        for kt in core_kts
        for name in ("p0_wait_mma_m0", "p0_wait_mma_m1")
    )
    producer_wait1 = sum(
        int(event(idx, kt, name)["cycles"])
        for kt in core_kts
        for name in ("p1_wait_mma_m0", "p1_wait_mma_m1")
    )
    producer_span0 = int(
        event(idx, core_kts[-1], "p0_prepare_and_issue_tma_b1")["end_raw"]
    ) - int(event(idx, core_kts[0], "p0_wait_mma_m0")["start_raw"])
    producer_span1 = int(
        event(idx, core_kts[-1], "p1_prepare_and_issue_tma_b0")["end_raw"]
    ) - int(event(idx, core_kts[0], "p1_wait_mma_m0")["start_raw"])
    summary = {
        "w2_commit_cadence_median": median(commit_cadence2),
        "w3_commit_cadence_median": median(commit_cadence3),
        "w2_wait_fraction": wait_total2 / max(span2, 1),
        "w3_wait_fraction": wait_total3 / max(span3, 1),
        "w0_reuse_wait_fraction": producer_wait0 / max(producer_span0, 1),
        "w1_reuse_wait_fraction": producer_wait1 / max(producer_span1, 1),
        "wait_a_median": median(
            [float(max(row["w2_wait_a"], row["w3_wait_a"])) for row in metric_rows]
        ),
        "wait_a_max": max(
            float(max(row["w2_wait_a"], row["w3_wait_a"])) for row in metric_rows
        ),
        "wait_b0_median": median(
            [float(max(row["w2_wait_b0"], row["w3_wait_b0"])) for row in metric_rows]
        ),
        "wait_b0_max": max(
            float(max(row["w2_wait_b0"], row["w3_wait_b0"])) for row in metric_rows
        ),
        "wait_b1_median": median(
            [float(max(row["w2_wait_b1"], row["w3_wait_b1"])) for row in metric_rows]
        ),
        "wait_b1_max": max(
            float(max(row["w2_wait_b1"], row["w3_wait_b1"])) for row in metric_rows
        ),
    }
    return metric_rows, summary


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_svg(
    rows: list[dict[str, object]],
    metrics: list[dict[str, object]],
    summary: dict[str, float],
    title: str,
) -> str:
    idx = index_rows(rows)
    meta = rows[0]
    k_start = int(meta["k_start"])
    core_count = int(meta["core_k_count"])
    core_kts = list(range(k_start, k_start + core_count))
    context_kts = list(range(k_start + core_count, k_start + int(meta["k_count"])))

    drawable: list[dict[str, object]] = []
    for kt in core_kts:
        for names in CORE_EVENTS.values():
            for name in names:
                drawable.append(event(idx, kt, name))
    for kt in context_kts:
        for name in (
            "p0_wait_mma_m0",
            "p0_wait_mma_m1",
            "p1_wait_mma_m0",
            "p1_wait_mma_m1",
        ):
            drawable.append(event(idx, kt, name))

    x0 = min(int(row["start_raw"]) for row in drawable)
    x1 = max(int(row["end_raw"]) for row in drawable)
    for kt in core_kts:
        for warp in (2, 3):
            x1 = max(x1, int(mma_observation(idx, kt, warp)["all_observed"]))
    span = x1 - x0

    width = 1840
    left = 225
    right = 55
    lane_top = 190
    lane_height = 104
    bar_y_offset = 22
    bar_height = 28
    lane_bottom = lane_top + 4 * lane_height
    legend_y = lane_bottom + 48
    cards_y = legend_y + 42
    table_y = cards_y + 132
    header_y = table_y + 34
    note_y = header_y + 33 + len(metrics) * 34 + 20
    height = max(1280, int(note_y + 145))
    plot_width = width - left - right
    scale = plot_width / max(span, 1)

    def xpos(raw: int | float) -> float:
        return left + (float(raw) - x0) * scale

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">'
        '<feDropShadow dx="0" dy="1" stdDeviation="1.5" '
        'flood-color="#0f172a" flood-opacity="0.10"/></filter>',
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        text(45, 48, title, size=27, weight=700),
        text(
            45,
            78,
            (
                f'16K BF16 GEMM · E7a dual-wide · input={meta["input"]} · '
                f'block={meta["block_idx"]} SM={meta["sm_id"]} · '
                f'tile_iter={meta["tile_iter"]} output=({meta["tile_m"]},{meta["tile_n"]})'
            ),
            size=14,
            fill="#475569",
        ),
        text(
            45,
            102,
            (
                f'core K64 stages k{core_kts[0]}–k{core_kts[-1]}; '
                f'k{context_kts[0]}–k{context_kts[-1]} retained only to observe '
                "producer dependency-pass evidence for core MMA through 3-stage reuse"
            ),
            size=13,
            fill="#64748b",
        ),
        text(45, 145, "Per-warp issue / wait timeline", size=18, weight=700),
        text(
            width - right,
            145,
            "clock64 cycles, normalized to first visible event",
            size=12,
            fill="#64748b",
            anchor="end",
        ),
    ]

    # Backgrounds first, then grids/guides, so the timing references remain
    # visible inside the lanes.
    for warp in range(4):
        y = lane_top + warp * lane_height
        out.append(
            f'<rect x="35" y="{y - 7:.2f}" width="{width - 70}" '
            f'height="{lane_height - 7}" rx="8" '
            f'fill="{"#ffffff" if warp % 2 == 0 else "#f1f5f9"}"/>'
        )
        out.append(
            text(left - 14, y + 40, WARP_LABELS[warp],
                 size=13, weight=650, anchor="end")
        )

    tick = nice_step(span)
    for relative in range(0, span + tick, tick):
        if relative > span:
            break
        x = xpos(x0 + relative)
        out.append(
            f'<line x1="{x:.2f}" y1="{lane_top - 20}" '
            f'x2="{x:.2f}" y2="{lane_bottom + 14}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        out.append(
            text(
                x,
                lane_top - 27,
                f"{relative:,}",
                size=11,
                fill="#64748b",
                anchor="middle",
            )
        )

    # Producer A-issue starts are readable stage anchors, not synchronization
    # boundaries.  They make ring-slot cadence visible without implying a CTA
    # wide phase boundary.
    for kt in core_kts:
        row = event(idx, kt, "p0_prepare_and_issue_tma_a")
        x = xpos(int(row["start_raw"]))
        slot = int(row["ring_stage"])
        out.append(
            f'<line x1="{x:.2f}" y1="{lane_top - 8}" '
            f'x2="{x:.2f}" y2="{lane_bottom + 6}" '
            f'stroke="{SLOT_COLORS[slot]}" stroke-width="1" '
            'stroke-dasharray="3,5" opacity="0.42"/>'
        )
        out.append(
            text(
                x + 4,
                lane_top - 6,
                f"k{kt}/s{slot}/p{row['phase']}",
                size=10,
                fill=SLOT_COLORS[slot],
            )
        )

    for row in drawable:
        name = str(row["event"])
        label, color = EVENT_STYLE[name]
        warp = int(row["warp"])
        kt = int(row["kt"])
        slot = int(row["ring_stage"])
        y = lane_top + warp * lane_height + bar_y_offset
        start = int(row["start_raw"])
        end = int(row["end_raw"])
        is_context = kt not in core_kts
        display = (
            f"k{kt} {label} | stage={slot} phase={row['phase']} | "
            f"{start - x0:,}–{end - x0:,} | {end - start:,} cycles"
        )
        if "wait_mma" in name:
            display += (
                f" | executed at k{kt}; observes MMA "
                f"k{int(row['related_kt'])} dependency pass"
            )
        out.append(
            rect(
                xpos(start),
                y,
                xpos(end) - xpos(start),
                bar_height,
                color,
                display,
                opacity=0.46 if is_context else 0.90,
                stroke=SLOT_COLORS[slot],
                dash="4,3" if is_context else "",
            )
        )
        pixel_width = xpos(end) - xpos(start)
        if pixel_width >= 34:
            label_text = f"k{kt}"
            if "issue_tma" in name:
                label_text += " TMA"
            elif "issue_mma" in name:
                label_text += " MMA"
            elif "commit" in name:
                label_text += " C"
            elif "wait_mma" in name:
                label_text = f"k{kt}←k{int(row['related_kt'])}"
            out.append(
                text(
                    xpos(start) + 4,
                    y + 19,
                    label_text,
                    size=9,
                    fill="#ffffff" if not is_context else "#334155",
                    weight=600,
                )
            )

    # Draw dashed software-observation connectors, not hardware activity bars.
    # Endpoint circles sit on the warp that actually passed the dependency.
    tma_colors = {"A": "#16a34a", "B0": "#0891b2", "B1": "#0f766e"}
    tma_lane = {"A": 0, "B0": 1, "B1": 0}
    tma_track = {"A": 0, "B0": 1, "B1": 2}
    tma_wait_names = {
        "A": ("c2_wait_tma_a", "c3_wait_tma_a"),
        "B0": ("c2_wait_tma_b0", "c3_wait_tma_b0"),
        "B1": ("c2_wait_tma_b1", "c3_wait_tma_b1"),
    }
    for kt in core_kts:
        for operand in ("A", "B0", "B1"):
            obs = tma_observation(idx, kt, operand)
            y = (
                lane_top
                + tma_lane[operand] * lane_height
                + bar_y_offset
                + bar_height
                + 6
                + tma_track[operand] * 6
            )
            color = tma_colors[operand]
            tooltip = (
                f"k{kt} {operand} TMA post-call → consumer wait-pass "
                f"observation | "
                f"issue_end={int(obs['issue_end']) - x0:,} | "
                f"first={int(obs['first_observed']) - x0:,} | "
                f"all={int(obs['all_observed']) - x0:,} | "
                f"signed post-issue gap={int(obs['post_issue_gap']):,} cycles | "
                f"issue-start bound={int(obs['issue_to_first_pass_bound']):,}"
            )
            if obs["post_issue_gap"] >= 0:
                out.append(
                    f'<line x1="{xpos(obs["issue_end"]):.2f}" y1="{y:.2f}" '
                    f'x2="{xpos(obs["first_observed"]):.2f}" y2="{y:.2f}" '
                    f'stroke="{color}" stroke-width="2" '
                    'stroke-dasharray="4,3" opacity="0.55"><title>'
                    f"{html.escape(tooltip)}</title></line>"
                )
            for wait_name in tma_wait_names[operand]:
                wait_row = event(idx, kt, wait_name)
                observer_warp = int(wait_row["warp"])
                observer_y = (
                    lane_top
                    + observer_warp * lane_height
                    + bar_y_offset
                    + bar_height
                    + 6
                    + tma_track[operand] * 6
                )
                observed = int(wait_row["end_raw"])
                first = observed == int(obs["first_observed"])
                observer_tip = (
                    f"k{kt} {operand}: W{observer_warp} consumer passed ready "
                    f"wait at {observed - x0:,} cycles"
                )
                marker_fill = color if first else "#ffffff"
                out.append(
                    f'<circle cx="{xpos(observed):.2f}" cy="{observer_y:.2f}" '
                    f'r="3.2" fill="{marker_fill}" '
                    f'stroke="{color}" stroke-width="1.8"><title>'
                    f"{html.escape(observer_tip)}</title></circle>"
                )

        for warp in (2, 3):
            obs = mma_observation(idx, kt, warp)
            slot = int(
                event(idx, kt, "p0_prepare_and_issue_tma_a")["ring_stage"]
            )
            y = (
                lane_top
                + warp * lane_height
                + bar_y_offset
                + bar_height
                + 30
                + slot * 6
            )
            color = "#7c3aed" if warp == 3 else "#2563eb"
            tooltip = (
                f"k{kt} W{warp} MMA commit post-call → producer reuse-wait-pass "
                f"observation at k{int(obs['reuse_kt'])} | "
                f"commit_end={int(obs['commit_end']) - x0:,} | first="
                f"{int(obs['first_observed']) - x0:,} | "
                f"signed post-commit gap={int(obs['post_commit_gap']):,} | "
                f"prepare-start bound="
                f"{int(obs['prepare_to_first_reuse_pass_bound']):,}"
            )
            if obs["post_commit_gap"] >= 0:
                out.append(
                    f'<line x1="{xpos(obs["commit_end"]):.2f}" y1="{y:.2f}" '
                    f'x2="{xpos(obs["first_observed"]):.2f}" y2="{y:.2f}" '
                    f'stroke="{color}" stroke-width="2" '
                    'stroke-dasharray="4,3" opacity="0.42"><title>'
                    f"{html.escape(tooltip)}</title></line>"
                )
            mblock = 0 if warp == 2 else 1
            reuse_names = (
                f"p0_wait_mma_m{mblock}",
                f"p1_wait_mma_m{mblock}",
            )
            for producer_warp, reuse_name in enumerate(reuse_names):
                reuse_row = event(idx, int(obs["reuse_kt"]), reuse_name)
                observed = int(reuse_row["end_raw"])
                observer_y = (
                    lane_top
                    + producer_warp * lane_height
                    + bar_y_offset
                    + bar_height
                    + 30
                    + slot * 6
                )
                first = observed == int(obs["first_observed"])
                observer_tip = (
                    f"k{kt} W{warp} MMA: W{producer_warp} passed M{mblock} "
                    f"reuse wait at k{int(obs['reuse_kt'])}, "
                    f"{observed - x0:,} cycles"
                )
                marker_fill = color if first else "#ffffff"
                out.append(
                    f'<circle cx="{xpos(observed):.2f}" cy="{observer_y:.2f}" '
                    f'r="3.2" fill="{marker_fill}" '
                    f'stroke="{color}" stroke-width="1.8"><title>'
                    f"{html.escape(observer_tip)}</title></circle>"
                )

    legend = [
        ("TMA prepare + issue", "#22c55e"),
        ("ready wait", "#f97316"),
        ("MMA prepare + issue", "#3b82f6"),
        ("commit", "#ec4899"),
        ("reuse wait", "#64748b"),
    ]
    lx = left
    for label, color in legend:
        out.append(
            f'<rect x="{lx:.2f}" y="{legend_y - 12}" width="14" height="14" '
            f'rx="2" fill="{color}"/>'
        )
        out.append(text(lx + 20, legend_y, label, size=12, fill="#475569"))
        lx += 58 + len(label) * 7
    out.append(
        text(
            width - right,
            legend_y,
            "dashed x-window on issuer lane; circles on observer lanes · filled/hollow = first/other dependency pass",
            size=11,
            fill="#64748b",
            anchor="end",
        )
    )

    card_width = (width - 90 - 3 * 16) / 4
    cards = [
        (
            "MMA commit cadence",
            f"W2 {summary['w2_commit_cadence_median']:.0f} / "
            f"W3 {summary['w3_commit_cadence_median']:.0f} cyc",
            f"median over {max(core_count - 1, 0)} core transitions",
        ),
        (
            "consumer ready-wait occupancy",
            f"W2 {100*summary['w2_wait_fraction']:.1f}% / "
            f"W3 {100*summary['w3_wait_fraction']:.1f}%",
            "measured wait-call spans / traced warp span",
        ),
        (
            "producer reuse-wait occupancy",
            f"W0 {100*summary['w0_reuse_wait_fraction']:.1f}% / "
            f"W1 {100*summary['w1_reuse_wait_fraction']:.1f}%",
            "measured wait-call spans / traced warp span",
        ),
        (
            "ready wait-call cycles",
            f"A {summary['wait_a_median']:.0f}/{summary['wait_a_max']:.0f} · "
            f"B0 {summary['wait_b0_median']:.0f}/{summary['wait_b0_max']:.0f} · "
            f"B1 {summary['wait_b1_median']:.0f}/{summary['wait_b1_max']:.0f}",
            "median/max; slower of W2/W3 in each stage",
        ),
    ]
    for index, (label, value, note) in enumerate(cards):
        x = 45 + index * (card_width + 16)
        out.append(
            f'<rect x="{x:.2f}" y="{cards_y:.2f}" width="{card_width:.2f}" '
            'height="86" rx="10" fill="#ffffff" stroke="#e2e8f0" '
            'filter="url(#shadow)"/>'
        )
        out.append(text(x + 16, cards_y + 24, label, size=12, fill="#64748b"))
        out.append(text(x + 16, cards_y + 51, value, size=18, weight=700))
        out.append(text(x + 16, cards_y + 73, note, size=10, fill="#64748b"))

    out.append(text(45, table_y, "Per-stage diagnostic metrics", size=18, weight=700))
    out.append(
        text(
            width - right,
            table_y,
            "* gaps/bounds use cross-warp software dependency-pass observations",
            size=11,
            fill="#64748b",
            anchor="end",
        )
    )
    table_left = 45
    table_right = width - right
    labels = [
        "kt/s/p",
        "TMA wait W2 A/B0/B1",
        "TMA wait W3 A/B0/B1",
        "reuse W0 M0/M1",
        "reuse W1 M0/M1",
        "TMA post-gap* A/B0/B1",
        "TMA start-bound* A/B0/B1",
        "MMA post-gap* W2/W3",
        "MMA prepare-bound* W2/W3",
        "commit Δ W2/W3",
    ]
    col_widths = [130, 165, 165, 145, 145, 215, 215, 175, 200, 145]
    positions = [table_left]
    for value in col_widths[:-1]:
        positions.append(positions[-1] + value)
    out.append(
        f'<rect x="{table_left}" y="{header_y - 22}" '
        f'width="{table_right - table_left}" height="31" rx="5" fill="#e2e8f0"/>'
    )
    for pos, label in zip(positions, labels):
        out.append(text(pos + 7, header_y, label, size=10, weight=650))

    for row_index, row in enumerate(metrics):
        y = header_y + 33 + row_index * 34
        out.append(
            f'<rect x="{table_left}" y="{y - 22}" '
            f'width="{table_right - table_left}" height="31" rx="4" '
            f'fill="{"#ffffff" if row_index % 2 == 0 else "#f1f5f9"}"/>'
        )
        values = [
            f"k{row['kt']}/s{row['ring_stage']}/p{row['phase']}",
            (
                f"{row['w2_wait_a']}/{row['w2_wait_b0']}/"
                f"{row['w2_wait_b1']}"
            ),
            (
                f"{row['w3_wait_a']}/{row['w3_wait_b0']}/"
                f"{row['w3_wait_b1']}"
            ),
            f"{row['w0_reuse_m0']}/{row['w0_reuse_m1']}",
            f"{row['w1_reuse_m0']}/{row['w1_reuse_m1']}",
            (
                f"{int(row['a_post_issue_gap']):+d}/"
                f"{int(row['b0_post_issue_gap']):+d}/"
                f"{int(row['b1_post_issue_gap']):+d}"
            ),
            (
                f"{row['a_issue_to_first_pass_bound']}/"
                f"{row['b0_issue_to_first_pass_bound']}/"
                f"{row['b1_issue_to_first_pass_bound']}"
            ),
            (
                f"{int(row['w2_post_commit_gap']):+d}/"
                f"{int(row['w3_post_commit_gap']):+d}"
            ),
            (
                f"{row['w2_prepare_to_first_reuse_pass_bound']}/"
                f"{row['w3_prepare_to_first_reuse_pass_bound']}"
            ),
            (
                f"{row['w2_commit_cadence'] or '—'}/"
                f"{row['w3_commit_cadence'] or '—'}"
            ),
        ]
        for pos, value in zip(positions, values):
            out.append(text(pos + 7, y, value, size=10))

    out.extend(
        [
            f'<rect x="45" y="{note_y}" width="{width - 100}" height="116" '
            'rx="10" fill="#fff7ed" stroke="#fed7aa"/>',
            text(65, note_y + 27, "How to read this", size=13,
                 weight=700, fill="#9a3412"),
            text(
                65,
                note_y + 51,
                (
                    "TMA issue-bar end and MMA commit-bar end are issuer post-call "
                    "stamps; ready/reuse wait ends are dependency-pass observations, "
                    "not exact hardware completion edges."
                ),
                size=11,
                fill="#7c2d12",
            ),
            text(
                65,
                note_y + 73,
                (
                    "* Signed post-gap is observer wait-end minus issuer post-call "
                    "stamp. A negative value is cross-warp ordering, not negative "
                    "hardware latency; the start/prepare bound is conservative."
                ),
                size=11,
                fill="#7c2d12",
            ),
            text(
                65,
                note_y + 95,
                (
                    "Wait bars are ordered residual wait-call durations. Dashed "
                    "x-windows are not hardware activity; clock64/trace stores perturb "
                    "this diagnostic binary, so do not use its runtime for TFLOP/s."
                ),
                size=11,
                fill="#7c2d12",
            ),
            "</svg>",
        ]
    )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument(
        "--title", default="B200 GEMM per-K-stage pipeline trace"
    )
    args = parser.parse_args()

    rows = read_trace(args.trace)
    metrics, summary = compute_metrics(rows)
    svg = build_svg(rows, metrics, summary, args.title)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(svg)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(args.metrics_csv, metrics)
    print(
        f"svg={args.svg} metrics={args.metrics_csv} "
        f"core=k{rows[0]['k_start']}.."
        f"k{int(rows[0]['k_start']) + int(rows[0]['core_k_count']) - 1}"
    )


if __name__ == "__main__":
    main()
