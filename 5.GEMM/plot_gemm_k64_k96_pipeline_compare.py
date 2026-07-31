#!/usr/bin/env python3
"""Draw same-scale actual clock64 traces for 3xK64 and 2xK96 pipelines."""

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
    "reuse": "#CBD5E1",
    "wait_a": "#FBBF24",
    "wait_b": "#60A5FA",
    "a": "#22C55E",
    "b0": "#06B6D4",
    "b1": "#14B8A6",
    "mma0": "#2563EB",
    "mma1": "#7C3AED",
    "commit": "#EC4899",
    "reuse_dep": "#DC2626",
}
LANES = {
    "W0 · A + B0 producer": 3,
    "W1 · B1 producer": 2,
    "W2 · C[:, 0:128] MMA": 1,
    "W3 · C[:, 128:256] MMA": 0,
}


def read_trace(path: Path) -> dict[tuple[int, str], tuple[int, int]]:
    rows: dict[tuple[int, str], tuple[int, int]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            kt = int(row["kt"])
            if kt >= 0:
                rows[(kt, row["event"])] = (
                    int(row["start_cycle"]),
                    int(row["end_cycle"]),
                )
    return rows


def bar(ax, y: float, start: float, end: float, color: str, label: str = "") -> None:
    width = max(1, end - start)
    ax.broken_barh(
        [(start, width)], (y - 0.31, 0.62), facecolor=color,
        edgecolor="#334155", linewidth=0.55, zorder=3,
    )
    if label and width >= 82:
        ax.text(
            start + width / 2, y, label, ha="center", va="center",
            fontsize=6.2,
            color="#1E293B" if color in {COLORS["wait_a"], COLORS["wait_b"]} else "white",
            fontweight="bold", clip_on=True, zorder=4,
        )


def arrow(ax, start, start_y, end, end_y, color, radius, dashed=False) -> None:
    ax.annotate(
        "", xy=(end, end_y), xytext=(start, start_y),
        arrowprops={
            "arrowstyle": "->", "color": color, "linewidth": 1.15,
            "linestyle": "--" if dashed else "-",
            "connectionstyle": f"arc3,rad={radius}",
        },
        zorder=5,
    )


def draw_trace(ax, trace, stages: int, tile_k: int, title: str, pass_label: str) -> float:
    kts = range(56, 64)
    first = min(
        trace[(kt, event)][0]
        for kt in kts for event in ("p0_wait_pipe0", "p1_wait_pipe1")
    )
    last = max(
        trace[(kt, event)][1]
        for kt in kts for event in ("c2_commit", "c3_commit")
    )

    for kt in kts:
        index = kt - kts.start
        w00, _ = trace[(kt, "p0_wait_pipe0")]
        _, w02 = trace[(kt, "p0_wait_pipe1")]
        a0, a1 = trace[(kt, "p0_issue_a")]
        b00, b01 = trace[(kt, "p0_issue_b0")]
        p10, p11 = trace[(kt, "p1_wait_pipe1")]
        b10, b11 = trace[(kt, "p1_issue_b1")]
        bar(ax, 3, w00 - first, w02 - first, COLORS["reuse"])
        bar(ax, 3, a0 - first, a1 - first, COLORS["a"], f"A{index}")
        bar(ax, 3, b00 - first, b01 - first, COLORS["b0"])
        bar(ax, 2, p10 - first, p11 - first, COLORS["reuse"])
        bar(ax, 2, b10 - first, b11 - first, COLORS["b1"], f"B1 {index}")

        for y, a_event, b_event, mma_event, commit_event, mma_color in (
            (1, "c2_wait_a", "c2_wait_b0", "c2_mma", "c2_commit", COLORS["mma0"]),
            (0, "c3_wait_a", "c3_wait_b1", "c3_mma", "c3_commit", COLORS["mma1"]),
        ):
            wa0, wa1 = trace[(kt, a_event)]
            wb0, wb1 = trace[(kt, b_event)]
            m0, m1 = trace[(kt, mma_event)]
            c0, c1 = trace[(kt, commit_event)]
            bar(ax, y, wa0 - first, wa1 - first, COLORS["wait_a"], "wait A")
            bar(ax, y, wb0 - first, wb1 - first, COLORS["wait_b"], "wait B")
            bar(ax, y, m0 - first, m1 - first, mma_color, f"MMA {index}")
            bar(ax, y, c0 - first, c1 - first, COLORS["commit"])

    # Show one representative set of ready and ring-reuse dependencies.
    dep_kt = kts.start + stages
    prev_kt = dep_kt - stages
    a_end = trace[(dep_kt, "p0_issue_a")][1] - first
    b0_end = trace[(dep_kt, "p0_issue_b0")][1] - first
    b1_end = trace[(dep_kt, "p1_issue_b1")][1] - first
    w2_mma = trace[(dep_kt, "c2_mma")][0] - first
    w3_mma = trace[(dep_kt, "c3_mma")][0] - first
    for values in (
        (a_end, 3.28, w2_mma, 1.30, COLORS["a"], 0.10),
        (b0_end, 3.28, w2_mma, 1.30, COLORS["b0"], -0.08),
        (a_end, 2.70, w3_mma, 0.30, COLORS["a"], 0.18),
        (b1_end, 2.30, w3_mma, 0.30, COLORS["b1"], -0.08),
    ):
        arrow(ax, *values, dashed=True)

    w2_commit = trace[(prev_kt, "c2_commit")][1] - first
    w3_commit = trace[(prev_kt, "c3_commit")][1] - first
    a_start = trace[(dep_kt, "p0_issue_a")][0] - first
    b1_start = trace[(dep_kt, "p1_issue_b1")][0] - first
    for values in (
        (w2_commit, 1.30, a_start, 2.70, COLORS["reuse_dep"], -0.13),
        (w3_commit, 0.30, a_start, 2.70, COLORS["reuse_dep"], 0.13),
        (w3_commit, 0.30, b1_start, 1.70, COLORS["reuse_dep"], -0.16),
    ):
        arrow(ax, *values)

    ax.set_yticks(list(LANES.values()), list(LANES.keys()))
    ax.set_ylim(-0.55, 3.55)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_xlabel("cycles relative to first observed event")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    ax.text(
        0.985, 0.94,
        f"{stages} SMEM stages × K={tile_k} · B-first wait · {pass_label}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        fontweight="bold", color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.94},
    )
    return last - first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k64", required=True, type=Path)
    parser.add_argument("--k96", required=True, type=Path)
    parser.add_argument("--k64-label", default="selected pass")
    parser.add_argument("--k96-label", default="selected pass")
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8.2), constrained_layout=True)
    k64_end = draw_trace(
        axes[0], read_trace(args.k64), 3, 64,
        "A. Canonical 3×K64 pipeline · actual clock64 trace", args.k64_label,
    )
    k96_end = draw_trace(
        axes[1], read_trace(args.k96), 2, 96,
        "B. Experimental 2×K96 pipeline · actual clock64 trace", args.k96_label,
    )
    xmax = max(k64_end, k96_end) + 120
    for ax in axes:
        ax.set_xlim(0, xmax)

    fig.suptitle(
        "Blackwell N-split GEMM · 3×K64 vs 2×K96 measured pipeline",
        fontsize=15, fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["reuse"], label="stage reuse wait"),
            Patch(facecolor=COLORS["wait_a"], label="consumer wait A"),
            Patch(facecolor=COLORS["wait_b"], label="consumer wait B0/B1"),
            Patch(facecolor=COLORS["a"], label="A TMA issue"),
            Patch(facecolor=COLORS["b0"], label="B0 TMA issue"),
            Patch(facecolor=COLORS["b1"], label="B1 TMA issue"),
            Patch(facecolor=COLORS["mma0"], label="W2 MMA"),
            Patch(facecolor=COLORS["mma1"], label="W3 MMA"),
            Patch(facecolor=COLORS["commit"], label="MMA commit"),
            Line2D([0], [0], color="#0F766E", linestyle="--", label="TMA ready → MMA"),
            Line2D([0], [0], color=COLORS["reuse_dep"], label="MMA done → slot reuse"),
        ],
        loc="outside lower center", ncol=6, frameon=False, fontsize=8,
    )
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150)
    print(f"svg={args.svg}")


if __name__ == "__main__":
    main()
