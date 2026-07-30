#!/usr/bin/env python3
"""Plot the current N-split clock64 trace as an issue/MMA pipeline figure."""

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
    "a": "#22C55E",
    "b0": "#06B6D4",
    "b1": "#14B8A6",
    "mma0": "#3B82F6",
    "mma1": "#8B5CF6",
    "commit": "#EC4899",
}


def read_trace(path: Path) -> list[dict[str, int | str]]:
    integer_fields = {
        "tile_iter",
        "linear_tile",
        "tile_m",
        "tile_n",
        "kt",
        "warp",
        "start_cycle",
        "end_cycle",
        "duration_cycle",
    }
    rows: list[dict[str, int | str]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, int | str] = dict(raw)
            for field in integer_fields:
                row[field] = int(raw[field])
            rows.append(row)
    return rows


def index_trace(
    rows: list[dict[str, int | str]],
) -> dict[tuple[int, str], dict[str, int | str]]:
    return {
        (int(row["kt"]), str(row["event"])): row
        for row in rows
        if int(row["kt"]) >= 0
    }


def interval(
    index: dict[tuple[int, str], dict[str, int | str]],
    kt: int,
    event: str,
) -> tuple[int, int]:
    row = index[(kt, event)]
    return int(row["start_cycle"]), int(row["end_cycle"])


def bar(ax, y: float, start: int, end: int, color: str, label: str = "") -> None:
    ax.broken_barh(
        [(start, end - start)],
        (y - 0.32, 0.64),
        facecolors=color,
        edgecolors="#334155",
        linewidth=0.55,
        zorder=3,
    )
    if label and end - start >= 72:
        ax.text(
            (start + end) / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color="white" if color != COLORS["wait"] else "#334155",
            fontweight="bold",
            clip_on=True,
            zorder=4,
        )


def plot_overview(
    ax,
    index: dict[tuple[int, str], dict[str, int | str]],
    kts: range,
) -> None:
    lanes = {
        "W0 · shared A + B0 producer": 3,
        "W1 · B1 producer": 2,
        "W2 · C[:, 0:128] consumer": 1,
        "W3 · C[:, 128:256] consumer": 0,
    }
    first = min(
        int(row["start_cycle"])
        for (kt, _), row in index.items()
        if kt in kts
    )
    last = max(
        int(row["end_cycle"])
        for (kt, _), row in index.items()
        if kt in kts
    )

    for kt in kts:
        w00, w01 = interval(index, kt, "p0_wait_pipe0")
        _, w02 = interval(index, kt, "p0_wait_pipe1")
        a0, a1 = interval(index, kt, "p0_issue_a")
        b00, b01 = interval(index, kt, "p0_issue_b0")
        p10, p11 = interval(index, kt, "p1_wait_pipe1")
        b10, b11 = interval(index, kt, "p1_issue_b1")
        c2a0, c2a1 = interval(index, kt, "c2_wait_a")
        _, c2b1 = interval(index, kt, "c2_wait_b0")
        m20, m21 = interval(index, kt, "c2_mma")
        cm20, cm21 = interval(index, kt, "c2_commit")
        c3a0, c3a1 = interval(index, kt, "c3_wait_a")
        _, c3b1 = interval(index, kt, "c3_wait_b1")
        m30, m31 = interval(index, kt, "c3_mma")
        cm30, cm31 = interval(index, kt, "c3_commit")

        bar(ax, 3, w00 - first, w02 - first, COLORS["wait"])
        bar(ax, 3, a0 - first, a1 - first, COLORS["a"], f"A {kt}")
        bar(ax, 3, b00 - first, b01 - first, COLORS["b0"])
        bar(ax, 2, p10 - first, p11 - first, COLORS["wait"])
        bar(ax, 2, b10 - first, b11 - first, COLORS["b1"], f"B1 {kt}")
        bar(ax, 1, c2a0 - first, c2b1 - first, COLORS["wait"])
        bar(ax, 1, m20 - first, m21 - first, COLORS["mma0"], f"MMA {kt}")
        bar(ax, 1, cm20 - first, cm21 - first, COLORS["commit"])
        bar(ax, 0, c3a0 - first, c3b1 - first, COLORS["wait"])
        bar(ax, 0, m30 - first, m31 - first, COLORS["mma1"], f"MMA {kt}")
        bar(ax, 0, cm30 - first, cm31 - first, COLORS["commit"])

    # One representative dependency connection.  Completion occurs no later
    # than the observed wait end; clock64 does not expose the exact TMA edge.
    kt = 59
    a0, a1 = interval(index, kt, "p0_issue_a")
    b00, b01 = interval(index, kt, "p0_issue_b0")
    b10, b11 = interval(index, kt, "p1_issue_b1")
    _, c2b1 = interval(index, kt, "c2_wait_b0")
    _, c3b1 = interval(index, kt, "c3_wait_b1")
    ax.annotate(
        "",
        xy=(c2b1 - first, 1.35),
        xytext=(a1 - first, 2.65),
        arrowprops=dict(arrowstyle="->", color=COLORS["a"], lw=1.1, ls="--"),
    )
    ax.annotate(
        "",
        xy=(c2b1 - first, 1.28),
        xytext=(b01 - first, 2.68),
        arrowprops=dict(arrowstyle="->", color=COLORS["b0"], lw=1.1, ls="--"),
    )
    ax.annotate(
        "",
        xy=(c3b1 - first, 0.32),
        xytext=(b11 - first, 1.68),
        arrowprops=dict(arrowstyle="->", color=COLORS["b1"], lw=1.1, ls="--"),
    )

    ax.set_yticks(list(lanes.values()), list(lanes.keys()))
    ax.set_xlim(0, last - first + 150)
    ax.set_ylim(-0.65, 3.7)
    ax.set_xlabel(f"cycles relative to trace cycle {first}")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_title(
        "Measured steady-state window: TMA for K stage i+2 overlaps MMA for stage i",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )


def plot_detail(
    ax,
    index: dict[tuple[int, str], dict[str, int | str]],
    kt: int,
) -> None:
    events = [
        ("p0_wait_pipe0", 5, COLORS["wait"], "wait p0"),
        ("p0_wait_pipe1", 5, "#94A3B8", "wait p1"),
        ("p0_issue_a", 4, COLORS["a"], "issue A"),
        ("p0_issue_b0", 3, COLORS["b0"], "issue B0"),
        ("p1_wait_pipe1", 2, COLORS["wait"], "wait p1"),
        ("p1_issue_b1", 2, COLORS["b1"], "issue B1"),
        ("c2_wait_a", 1, COLORS["wait"], "wait A"),
        ("c2_wait_b0", 1, "#94A3B8", "wait B0"),
        ("c2_mma", 1, COLORS["mma0"], "W2 MMA"),
        ("c3_wait_a", 0, COLORS["wait"], "wait A"),
        ("c3_wait_b1", 0, "#94A3B8", "wait B1"),
        ("c3_mma", 0, COLORS["mma1"], "W3 MMA"),
    ]
    first = min(interval(index, kt, event)[0] for event, *_ in events)
    last = max(interval(index, kt, event)[1] for event, *_ in events)
    for event, y, color, label in events:
        start, end = interval(index, kt, event)
        bar(ax, y, start - first, end - first, color, label)

    a0, a1 = interval(index, kt, "p0_issue_a")
    b00, b01 = interval(index, kt, "p0_issue_b0")
    b10, b11 = interval(index, kt, "p1_issue_b1")
    m20, _ = interval(index, kt, "c2_mma")
    m30, _ = interval(index, kt, "c3_mma")

    for issue_end, mma_start, y, color in [
        (a1, m20, 4, COLORS["a"]),
        (b01, m20, 3, COLORS["b0"]),
        (b11, m30, 2, COLORS["b1"]),
    ]:
        ax.plot(
            [issue_end - first, mma_start - first],
            [y - 0.42, y - 0.42],
            color=color,
            linestyle="--",
            linewidth=1.4,
        )

    ax.text(
        (a1 + m20) / 2 - first,
        3.43,
        f"A issue → W2 MMA: {m20 - a0} cyc",
        ha="center",
        fontsize=8,
        color="#166534",
    )
    ax.text(
        (b11 + m30) / 2 - first,
        1.43,
        f"B1 issue end → W3 MMA: {m30 - b11} cyc",
        ha="center",
        fontsize=8,
        color="#0F766E",
    )

    labels = {
        5: "W0 stage reuse",
        4: "W0 A TMA",
        3: "W0 B0 TMA",
        2: "W1 B1 TMA",
        1: "W2 waits + MMA",
        0: "W3 waits + MMA",
    }
    ax.set_yticks(list(labels), [labels[y] for y in labels])
    ax.set_xlim(0, last - first + 120)
    ax.set_ylim(-0.65, 5.65)
    ax.set_xlabel(f"cycles relative to kt={kt} first observed event")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_title(
        f"One K64 stage in detail (kt={kt}): issue span is not TMA transfer duration",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument("--detail-kt", type=int, default=59)
    args = parser.parse_args()

    rows = read_trace(args.trace)
    index = index_trace(rows)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15.5, 9.0),
        gridspec_kw={"height_ratios": [1.05, 1.0]},
        constrained_layout=True,
    )
    plot_overview(axes[0], index, range(56, 64))
    plot_detail(axes[1], index, args.detail_kt)
    fig.suptitle(
        "Blackwell N-split GEMM · A/B TMA and two-consumer MMA pipeline",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["wait"], label="barrier wait"),
            Patch(facecolor=COLORS["a"], label="A 256×64 TMA issue"),
            Patch(facecolor=COLORS["b0"], label="B0 64×128 TMA issue"),
            Patch(facecolor=COLORS["b1"], label="B1 64×128 TMA issue"),
            Patch(facecolor=COLORS["mma0"], label="W2 MMA issue span"),
            Patch(facecolor=COLORS["mma1"], label="W3 MMA issue span"),
        ],
        loc="outside lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg")
    print(f"svg={args.svg}")


if __name__ == "__main__":
    main()
