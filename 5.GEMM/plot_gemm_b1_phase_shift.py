#!/usr/bin/env python3
"""Draw the measured baseline and inferred B1 phase-shift pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLORS = {
    "wait": "#CBD5E1",
    "sleep": "#F59E0B",
    "a": "#22C55E",
    "b0": "#06B6D4",
    "b1": "#14B8A6",
    "mma0": "#3B82F6",
    "mma1": "#8B5CF6",
}


def read_trace(path: Path) -> dict[tuple[int, str], tuple[int, int]]:
    result: dict[tuple[int, str], tuple[int, int]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            result[(int(row["kt"]), row["event"])] = (
                int(row["start_cycle"]),
                int(row["end_cycle"]),
            )
    return result


def bar(
    ax,
    y: float,
    start: float,
    width: float,
    color: str,
    label: str = "",
    height: float = 0.58,
) -> None:
    ax.broken_barh(
        [(start, width)],
        (y - height / 2, height),
        facecolor=color,
        edgecolor="#334155",
        linewidth=0.6,
        zorder=3,
    )
    if label and width >= 65:
        ax.text(
            start + width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            color="white" if color != COLORS["wait"] else "#334155",
            fontweight="bold",
            zorder=4,
        )


def measured_panel(ax, trace: dict[tuple[int, str], tuple[int, int]], kt: int) -> None:
    events = [
        ("p0_wait_pipe0", 5, COLORS["wait"], ""),
        ("p0_wait_pipe1", 5, "#94A3B8", ""),
        ("p0_issue_a", 4, COLORS["a"], "A"),
        ("p0_issue_b0", 3, COLORS["b0"], "B0"),
        ("p1_wait_pipe1", 2, COLORS["wait"], "reuse wait"),
        ("p1_issue_b1", 2, COLORS["b1"], "B1"),
        ("c2_wait_a", 1, COLORS["wait"], ""),
        ("c2_wait_b0", 1, "#94A3B8", ""),
        ("c2_mma", 1, COLORS["mma0"], "W2 MMA"),
        ("c3_wait_a", 0, COLORS["wait"], ""),
        ("c3_wait_b1", 0, "#94A3B8", ""),
        ("c3_mma", 0, COLORS["mma1"], "W3 MMA"),
    ]
    first = min(trace[(kt, event)][0] for event, *_ in events)
    last = max(trace[(kt, event)][1] for event, *_ in events)
    for event, y, color, label in events:
        start, end = trace[(kt, event)]
        bar(ax, y, start - first, end - start, color, label)

    b1_end = trace[(kt, "p1_issue_b1")][1]
    mma_start = trace[(kt, "c3_mma")][0]
    ax.annotate(
        "",
        xy=(mma_start - first, 1.62),
        xytext=(b1_end - first, 1.62),
        arrowprops=dict(arrowstyle="<->", color="#0F766E", lw=1.2),
    )
    ax.text(
        (b1_end + mma_start) / 2 - first,
        1.75,
        f"B1 issue end → W3 MMA = {mma_start - b1_end} cyc",
        ha="center",
        fontsize=8,
        color="#0F766E",
    )
    ax.set_yticks(
        range(6),
        ["W3 waits + MMA", "W2 waits + MMA", "W1 B1", "W0 B0", "W0 A", "W0 reuse"],
    )
    ax.set_xlim(0, last - first + 100)
    ax.set_ylim(-0.55, 5.55)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_xlabel(f"cycles relative to the first kt={kt} event")
    ax.set_title(
        "A. Measured baseline stage (clock64 trace)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )


def cadence_panel(ax) -> None:
    # Effective periods are inferred from throughput relative to the measured
    # baseline cadence (~1050 cycles). They are not a second clock64 trace.
    cases = [
        ("delay 0", 0, 1050.0, 1856.453, 5.0),
        ("delay 96", 96, 1050.0 * 1856.335 / 1860.958, 1860.958, 3.1),
        ("delay 512", 512, 1050.0 * 1856.453 / 1170.017, 1170.017, 1.2),
    ]
    stage_count = 5
    for name, delay, period, tflops, y in cases:
        for stage in range(stage_count):
            origin = stage * period
            if delay:
                bar(ax, y, origin, delay, COLORS["sleep"], "", height=0.5)
            bar(ax, y, origin + delay, 105, COLORS["b1"], f"B1 {stage}", height=0.5)
            # A consumer is shown two steady-state stage slots downstream. The
            # precise delayed-kernel dependency edge was not traced.
            mma_start = origin + 2 * period
            bar(ax, y - 0.62, mma_start, 570, COLORS["mma1"], f"MMA {stage}", height=0.46)
    ax.axvspan(0, 0, color=COLORS["sleep"])
    ax.set_yticks(
        [5.0, 4.38, 3.1, 2.48, 1.2, 0.58],
        [
            "delay 0 · W1 B1\n1050 cyc/stage · 1856 TF/s",
            "delay 0 · W3 MMA",
            "delay 96 · W1 B1\n1047 cyc/stage · 1861 TF/s",
            "delay 96 · W3 MMA",
            "delay 512 · W1 B1\n1666 cyc/stage · 1170 TF/s",
            "delay 512 · W3 MMA",
        ],
    )
    ax.set_xlim(0, 8900)
    ax.set_ylim(0.15, 5.65)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_xlabel("schematic steady-state cycles")
    ax.set_title(
        "B. Repeated per-stage shift and measured effective cadence "
        "(schematic; period inferred from throughput)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        4300,
        5.55,
        "96 cycles mostly fits inside existing overlap; stage cadence is unchanged",
        ha="center",
        va="top",
        fontsize=9,
        color="#92400E",
    )
    ax.text(
        5850,
        1.82,
        "512 cycles repeats at every K64 stage → B1 producer cadence expands",
        ha="center",
        fontsize=9,
        color="#92400E",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--detail-kt", type=int, default=59)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.5), constrained_layout=True)
    measured_panel(axes[0], read_trace(args.trace), args.detail_kt)
    cadence_panel(axes[1])
    fig.suptitle(
        "Blackwell N-split GEMM · effect of delaying the B1 TMA producer",
        fontsize=15,
        fontweight="bold",
    )
    axes[0].legend(
        handles=[
            Patch(facecolor=COLORS["wait"], label="barrier/reuse wait"),
            Patch(facecolor=COLORS["sleep"], label="inserted NANOSLEEP"),
            Patch(facecolor=COLORS["a"], label="A 256×64 TMA issue"),
            Patch(facecolor=COLORS["b0"], label="B0 64×128 TMA issue"),
            Patch(facecolor=COLORS["b1"], label="B1 64×128 TMA issue"),
            Patch(facecolor=COLORS["mma0"], label="W2 MMA"),
            Patch(facecolor=COLORS["mma1"], label="W3 MMA"),
        ],
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        fontsize=7.5,
    )
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150)
    print(f"svg={args.svg}")


if __name__ == "__main__":
    main()
