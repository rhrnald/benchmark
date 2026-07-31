#!/usr/bin/env python3
"""Plot an actual alternating-K clock64 trace with dependency arrows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLORS = {
    "reuse0": "#CBD5E1",
    "reuse1": "#94A3B8",
    "a0": "#22C55E",
    "a1": "#15803D",
    "b0": "#06B6D4",
    "b1": "#14B8A6",
    "wait_b": "#60A5FA",
    "wait_a0": "#FBBF24",
    "wait_a1": "#F97316",
    "mma0": "#2563EB",
    "mma1": "#7C3AED",
    "commit": "#EC4899",
    "ready_dep": "#0F766E",
    "reuse_dep": "#DC2626",
}

LANES = {
    "W0 · A0/A1 + B0 producer": 3,
    "W1 · B1 producer": 2,
    "W2 · C[:, 0:128] MMA": 1,
    "W3 · C[:, 128:256] MMA": 0,
}


def read_trace(
    path: Path,
) -> tuple[
    dict[tuple[int, str], tuple[int, int]],
    dict[int, dict[str, int]],
]:
    events: dict[tuple[int, str], tuple[int, int]] = {}
    metadata: dict[int, dict[str, int]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            logical = int(row["logical_stage"])
            events[(logical, row["event"])] = (
                int(row["start_cycle"]),
                int(row["end_cycle"]),
            )
            metadata[logical] = {
                "stage_epoch": int(row["stage_epoch"]),
                "k64_cursor": int(row["k64_cursor"]),
                "stage_k": int(row["stage_k"]),
                "smem_slot": int(row["smem_slot"]),
            }
    return events, metadata


def bar(
    ax,
    y: float,
    start: float,
    end: float,
    color: str,
    label: str = "",
    *,
    edge: str = "#334155",
) -> None:
    width = max(1.0, end - start)
    ax.broken_barh(
        [(start, width)],
        (y - 0.31, 0.62),
        facecolor=color,
        edgecolor=edge,
        linewidth=0.55,
        zorder=3,
    )
    if label and width >= 100:
        light = color in {
            COLORS["reuse0"],
            COLORS["wait_b"],
            COLORS["wait_a0"],
        }
        ax.text(
            start + width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6.3,
            color="#1E293B" if light else "white",
            fontweight="bold",
            clip_on=True,
            zorder=4,
        )


def get(
    events: dict[tuple[int, str], tuple[int, int]],
    logical: int,
    event: str,
) -> tuple[int, int] | None:
    return events.get((logical, event))


def arrow(ax, start, end, color, radius, linestyle="--", linewidth=1.1):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "connectionstyle": f"arc3,rad={radius}",
        },
        zorder=6,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    events, metadata = read_trace(args.csv)
    logicals = sorted(metadata)
    first = min(start for start, _ in events.values())
    last = max(end for _, end in events.values())

    fig, ax = plt.subplots(figsize=(17, 6.7), constrained_layout=True)

    for logical in logicals:
        meta = metadata[logical]
        short = f"L{logical} K{meta['stage_k']} s{meta['smem_slot']}"
        for event, color, label in (
            ("p0_wait_pipe0", COLORS["reuse0"], ""),
            ("p0_wait_pipe1", COLORS["reuse1"], ""),
            ("p0_issue_a0", COLORS["a0"], f"A0 {short}"),
            ("p0_issue_a1", COLORS["a1"], f"A1 {short}"),
            ("p0_issue_b0", COLORS["b0"], f"B0 {short}"),
        ):
            interval = get(events, logical, event)
            if interval:
                bar(
                    ax,
                    3,
                    interval[0] - first,
                    interval[1] - first,
                    color,
                    label,
                )
        for event, color, label in (
            ("p1_wait_pipe1", COLORS["reuse1"], ""),
            ("p1_issue_b1", COLORS["b1"], f"B1 {short}"),
        ):
            interval = get(events, logical, event)
            if interval:
                bar(
                    ax,
                    2,
                    interval[0] - first,
                    interval[1] - first,
                    color,
                    label,
                )

        for y, prefix, mma_color in (
            (1, "c2", COLORS["mma0"]),
            (0, "c3", COLORS["mma1"]),
        ):
            for suffix, color, label in (
                ("wait_b0" if prefix == "c2" else "wait_b1",
                 COLORS["wait_b"], "wait B"),
                ("wait_a0", COLORS["wait_a0"], "wait A0"),
                ("wait_a1", COLORS["wait_a1"], "wait A1"),
                ("mma", mma_color, f"MMA {short}"),
                ("commit", COLORS["commit"], ""),
            ):
                interval = get(events, logical, f"{prefix}_{suffix}")
                if interval:
                    bar(
                        ax,
                        y,
                        interval[0] - first,
                        interval[1] - first,
                        color,
                        label,
                    )

    # Draw exact observed dependencies for the third logical stage. The TMA
    # completion moment is not directly timestamped; the corresponding wait
    # end is the first observed completion point.
    dep = logicals[2]
    prev = dep - 2
    for issue_event, wait_event, producer_y, consumer_y, color, radius in (
        ("p0_issue_b0", "c2_wait_b0", 3.30, 1.30, COLORS["b0"], -0.08),
        ("p1_issue_b1", "c3_wait_b1", 2.30, 0.30, COLORS["b1"], 0.08),
        ("p0_issue_a0", "c2_wait_a0", 3.30, 1.30, COLORS["a0"], 0.12),
        ("p0_issue_a1", "c2_wait_a1", 2.70, 1.30, COLORS["a1"], -0.12),
    ):
        issue = get(events, dep, issue_event)
        wait = get(events, dep, wait_event)
        if issue and wait:
            arrow(
                ax,
                (issue[1] - first, producer_y),
                (wait[1] - first, consumer_y),
                color,
                radius,
            )

    # The same SMEM slot is reused every two logical stages. A producer cannot
    # overwrite it until the previous consumers' completion barriers resolve.
    for commit_event, issue_event, start_y, end_y, radius in (
        ("c2_commit", "p0_issue_a0", 1.30, 2.70, -0.11),
        ("c3_commit", "p0_issue_a0", 0.30, 2.70, 0.11),
        ("c3_commit", "p1_issue_b1", 0.30, 1.70, -0.15),
    ):
        commit = get(events, prev, commit_event)
        issue = get(events, dep, issue_event)
        if commit and issue:
            arrow(
                ax,
                (commit[1] - first, start_y),
                (issue[0] - first, end_y),
                COLORS["reuse_dep"],
                radius,
                linestyle="-",
                linewidth=1.25,
            )

    ax.set_yticks(list(LANES.values()), list(LANES.keys()))
    ax.set_ylim(-0.58, 3.62)
    ax.set_xlim(0, last - first + 140)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_xlabel("actual SM clock cycles relative to first recorded event")
    ax.set_title(
        "Alternating K64/K128 GEMM · actual B200 clock64 pipeline trace",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.995,
        1.02,
        "block 0 · persistent tile_iter 8 · logical stages 56–63 · B-first",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#475569",
    )

    fig.legend(
        handles=[
            Patch(facecolor=COLORS["reuse0"], label="SMEM reuse wait"),
            Patch(facecolor=COLORS["a0"], label="A slab0 TMA issue"),
            Patch(facecolor=COLORS["a1"], label="A slab1 TMA issue"),
            Patch(facecolor=COLORS["b0"], label="B0 TMA issue"),
            Patch(facecolor=COLORS["b1"], label="B1 TMA issue"),
            Patch(facecolor=COLORS["wait_b"], label="consumer wait B"),
            Patch(facecolor=COLORS["wait_a0"], label="consumer wait A0"),
            Patch(facecolor=COLORS["wait_a1"], label="consumer wait A1"),
            Patch(facecolor=COLORS["mma0"], label="W2 MMA issue"),
            Patch(facecolor=COLORS["mma1"], label="W3 MMA issue"),
            Patch(facecolor=COLORS["commit"], label="MMA commit issue"),
            Line2D(
                [0],
                [0],
                color=COLORS["ready_dep"],
                linestyle="--",
                label="TMA issue → observed ready",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["reuse_dep"],
                label="MMA commit → slot-reuse issue",
            ),
        ],
        loc="outside lower center",
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.015,
        "clock64 brackets instruction issue and blocking waits. A wait end is "
        "the first observed TMA-completion point; commit marks barrier issue, "
        "not tensor-core completion.",
        ha="center",
        fontsize=8.2,
        color="#475569",
    )

    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=160)
    print(f"svg={args.svg}")


if __name__ == "__main__":
    main()
