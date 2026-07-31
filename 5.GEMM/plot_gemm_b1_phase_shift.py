#!/usr/bin/env python3
"""Draw a full steady-state pipeline view for measured and shifted B1 timing."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLORS = {
    "wait": "#CBD5E1",
    "wait2": "#94A3B8",
    "sleep": "#F59E0B",
    "a": "#22C55E",
    "b0": "#06B6D4",
    "b1": "#14B8A6",
    "mma0": "#3B82F6",
    "mma1": "#8B5CF6",
    "commit": "#EC4899",
}
LANES = {
    "W0 · A + B0 producer": 3,
    "W1 · B1 producer": 2,
    "W2 · C[:, 0:128] MMA": 1,
    "W3 · C[:, 128:256] MMA": 0,
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
    end: float,
    color: str,
    label: str = "",
) -> None:
    width = max(1.0, end - start)
    ax.broken_barh(
        [(start, width)],
        (y - 0.31, 0.62),
        facecolor=color,
        edgecolor="#334155",
        linewidth=0.55,
        zorder=3,
    )
    if label and width >= 72:
        ax.text(
            start + width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6.7,
            color="white" if color not in {COLORS["wait"], COLORS["sleep"]} else "#334155",
            fontweight="bold",
            clip_on=True,
            zorder=4,
        )


def setup_axis(ax, title: str, xmax: float) -> None:
    ax.set_yticks(list(LANES.values()), list(LANES.keys()))
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_xlabel("cycles")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")


def draw_measured(
    ax,
    trace: dict[tuple[int, str], tuple[int, int]],
    kts: range,
) -> None:
    first = min(trace[(kt, event)][0] for kt in kts for event in ("p0_wait_pipe0", "p1_wait_pipe1"))
    last = max(trace[(kt, event)][1] for kt in kts for event in ("c2_commit", "c3_commit"))
    for kt in kts:
        stage = kt - kts.start
        w00, _ = trace[(kt, "p0_wait_pipe0")]
        _, w02 = trace[(kt, "p0_wait_pipe1")]
        a0, a1 = trace[(kt, "p0_issue_a")]
        b00, b01 = trace[(kt, "p0_issue_b0")]
        p10, p11 = trace[(kt, "p1_wait_pipe1")]
        b10, b11 = trace[(kt, "p1_issue_b1")]
        c2a0, _ = trace[(kt, "c2_wait_a")]
        _, c2b1 = trace[(kt, "c2_wait_b0")]
        m20, m21 = trace[(kt, "c2_mma")]
        cm20, cm21 = trace[(kt, "c2_commit")]
        c3a0, _ = trace[(kt, "c3_wait_a")]
        _, c3b1 = trace[(kt, "c3_wait_b1")]
        m30, m31 = trace[(kt, "c3_mma")]
        cm30, cm31 = trace[(kt, "c3_commit")]

        bar(ax, 3, w00 - first, w02 - first, COLORS["wait"])
        bar(ax, 3, a0 - first, a1 - first, COLORS["a"], f"A {stage}")
        bar(ax, 3, b00 - first, b01 - first, COLORS["b0"])
        bar(ax, 2, p10 - first, p11 - first, COLORS["wait"])
        bar(ax, 2, b10 - first, b11 - first, COLORS["b1"], f"B1 {stage}")
        bar(ax, 1, c2a0 - first, c2b1 - first, COLORS["wait"])
        bar(ax, 1, m20 - first, m21 - first, COLORS["mma0"], f"MMA {stage}")
        bar(ax, 1, cm20 - first, cm21 - first, COLORS["commit"])
        bar(ax, 0, c3a0 - first, c3b1 - first, COLORS["wait"])
        bar(ax, 0, m30 - first, m31 - first, COLORS["mma1"], f"MMA {stage}")
        bar(ax, 0, cm30 - first, cm31 - first, COLORS["commit"])

    # Show only one representative stage so the steady-state pipeline remains
    # readable. TMA completion is not directly traced; the arrow head is the
    # observed consumer wait completion / MMA start.
    dep_kt = kts.start + 3
    a_end = trace[(dep_kt, "p0_issue_a")][1] - first
    b0_end = trace[(dep_kt, "p0_issue_b0")][1] - first
    b1_end = trace[(dep_kt, "p1_issue_b1")][1] - first
    w2_ready = trace[(dep_kt, "c2_mma")][0] - first
    w3_ready = trace[(dep_kt, "c3_mma")][0] - first
    for start, start_y, end, end_y, color, radius in (
        (a_end, 2.68, w2_ready, 1.31, COLORS["a"], 0.14),
        (b0_end, 2.68, w2_ready, 1.31, COLORS["b0"], -0.10),
        (b1_end, 1.68, w3_ready, 0.31, COLORS["b1"], 0.12),
    ):
        ax.annotate(
            "",
            xy=(end, end_y),
            xytext=(start, start_y),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": 1.35,
                "linestyle": "--",
                "connectionstyle": f"arc3,rad={radius}",
            },
            zorder=5,
        )
    ax.text(
        (b0_end + w2_ready) / 2,
        2.18,
        "representative data-ready dependencies (stage 3)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#475569",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=6,
    )

    setup_axis(
        ax,
        "A. Baseline · measured clock64 trace, eight consecutive K64 stages",
        last - first + 120,
    )


def median_interval(
    trace: dict[tuple[int, str], tuple[int, int]],
    kts: range,
    event: str,
) -> tuple[float, float]:
    anchors = [trace[(kt, "p1_issue_b1")][0] for kt in kts]
    starts = [trace[(kt, event)][0] - anchor for kt, anchor in zip(kts, anchors)]
    ends = [trace[(kt, event)][1] - anchor for kt, anchor in zip(kts, anchors)]
    return statistics.median(starts), statistics.median(ends)


def draw_modeled(
    ax,
    trace: dict[tuple[int, str], tuple[int, int]],
    template_kts: range,
    delay: int,
    period: float,
    tflops: float,
    stages: int,
    panel: str,
) -> None:
    intervals = {
        event: median_interval(trace, template_kts, event)
        for event in (
            "p0_wait_pipe0",
            "p0_wait_pipe1",
            "p0_issue_a",
            "p0_issue_b0",
            "p1_wait_pipe1",
            "p1_issue_b1",
            "c2_wait_a",
            "c2_wait_b0",
            "c2_mma",
            "c2_commit",
            "c3_wait_a",
            "c3_wait_b1",
            "c3_mma",
            "c3_commit",
        )
    }
    raw_first = min(intervals["p0_wait_pipe0"][0], intervals["p1_wait_pipe1"][0])
    shift = -raw_first

    for stage in range(stages):
        origin = shift + stage * period

        w0_start = origin + intervals["p0_wait_pipe0"][0]
        w0_end = origin + intervals["p0_wait_pipe1"][1]
        bar(ax, 3, w0_start, w0_end, COLORS["wait"])
        for event, color, label in (
            ("p0_issue_a", COLORS["a"], f"A {stage}"),
            ("p0_issue_b0", COLORS["b0"], ""),
        ):
            start, end = intervals[event]
            bar(ax, 3, origin + start, origin + end, color, label)

        p1_start, _ = intervals["p1_wait_pipe1"]
        bar(ax, 2, origin + p1_start, origin, COLORS["wait"])
        bar(ax, 2, origin, origin + delay, COLORS["sleep"], "sleep")
        b1_duration = intervals["p1_issue_b1"][1] - intervals["p1_issue_b1"][0]
        bar(
            ax,
            2,
            origin + delay,
            origin + delay + b1_duration,
            COLORS["b1"],
            f"B1 {stage}",
        )

        for y, wait_a, wait_b, mma, commit, color in (
            (1, "c2_wait_a", "c2_wait_b0", "c2_mma", "c2_commit", COLORS["mma0"]),
            (0, "c3_wait_a", "c3_wait_b1", "c3_mma", "c3_commit", COLORS["mma1"]),
        ):
            bar(
                ax,
                y,
                origin + intervals[wait_a][0],
                origin + intervals[wait_b][1],
                COLORS["wait"],
            )
            start, end = intervals[mma]
            bar(ax, y, origin + start, origin + end, color, f"MMA {stage}")
            start, end = intervals[commit]
            bar(ax, y, origin + start, origin + end, COLORS["commit"])

    xmax = shift + (stages - 1) * period + max(
        intervals["c2_commit"][1], intervals["c3_commit"][1], delay + b1_duration
    )
    setup_axis(
        ax,
        f"{panel}. B1 delay {delay} · modeled full pipeline "
        f"({period:.0f} cyc/stage, {tflops:.0f} TFLOP/s)",
        xmax + 120,
    )
    ax.text(
        0.995,
        0.04,
        "median baseline event offsets + measured effective cadence",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#64748B",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    trace = read_trace(args.trace)
    template_kts = range(56, 64)
    fig, axes = plt.subplots(3, 1, figsize=(16, 11.2), constrained_layout=True)
    draw_measured(axes[0], trace, template_kts)
    draw_modeled(
        axes[1],
        trace,
        template_kts,
        delay=96,
        period=1050.0 * 1856.335 / 1860.958,
        tflops=1860.958,
        stages=8,
        panel="B",
    )
    draw_modeled(
        axes[2],
        trace,
        template_kts,
        delay=512,
        period=1050.0 * 1856.453 / 1170.017,
        tflops=1170.017,
        stages=8,
        panel="C",
    )
    fig.suptitle(
        "Blackwell N-split GEMM · full TMA/MMA pipeline under B1 phase shift",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["wait"], label="barrier/reuse wait"),
            Patch(facecolor=COLORS["sleep"], label="inserted NANOSLEEP"),
            Patch(facecolor=COLORS["a"], label="A 256×64 TMA issue"),
            Patch(facecolor=COLORS["b0"], label="B0 64×128 TMA issue"),
            Patch(facecolor=COLORS["b1"], label="B1 64×128 TMA issue"),
            Patch(facecolor=COLORS["mma0"], label="W2 MMA"),
            Patch(facecolor=COLORS["mma1"], label="W3 MMA"),
            Patch(facecolor=COLORS["commit"], label="MMA commit"),
        ],
        loc="outside lower center",
        ncol=8,
        frameon=False,
        fontsize=8,
    )
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150)
    print(f"svg={args.svg}")


if __name__ == "__main__":
    main()
