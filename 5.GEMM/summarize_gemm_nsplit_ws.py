#!/usr/bin/env python3
"""Summarize the matched N-split B-collector experiment."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


VARIANTS = (
    "e7a_exact",
    "nsplit_exact",
    "nsplit_static_control",
    "nsplit_ws_b01",
)
INPUTS = ("random", "random-signed8")
PASSES = range(1, 5)


def load_value(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one result row, got {len(rows)}")
    return float(rows[0]["event_TFLOPS"])


def paired_delta(values: list[float], control: list[float]) -> float:
    return statistics.mean(
        (value / base - 1.0) * 100.0
        for value, base in zip(values, control)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    csv_dir = result_dir / "csv"

    samples: dict[tuple[str, str], list[float]] = {}
    for input_name in INPUTS:
        for variant in VARIANTS:
            samples[(input_name, variant)] = [
                load_value(csv_dir / f"{variant}_{input_name}_p{pass_idx}.csv")
                for pass_idx in PASSES
            ]

    aggregate_path = result_dir / "aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "input",
                "variant",
                "sample_1",
                "sample_2",
                "sample_3",
                "sample_4",
                "mean_tflops",
                "sample_sd_tflops",
                "vs_e7a_ratio_of_means_percent",
                "vs_e7a_mean_paired_percent",
                "vs_nsplit_ratio_of_means_percent",
                "vs_nsplit_mean_paired_percent",
                "vs_static_ratio_of_means_percent",
                "vs_static_mean_paired_percent",
            )
        )
        for input_name in INPUTS:
            e7a = samples[(input_name, "e7a_exact")]
            nsplit = samples[(input_name, "nsplit_exact")]
            static_control = samples[(input_name, "nsplit_static_control")]
            for variant in VARIANTS:
                values = samples[(input_name, variant)]
                mean = statistics.mean(values)
                writer.writerow(
                    (
                        input_name,
                        variant,
                        *(f"{value:.6f}" for value in values),
                        f"{mean:.6f}",
                        f"{statistics.stdev(values):.6f}",
                        f"{(mean / statistics.mean(e7a) - 1.0) * 100.0:.6f}",
                        f"{paired_delta(values, e7a):.6f}",
                        f"{(mean / statistics.mean(nsplit) - 1.0) * 100.0:.6f}",
                        f"{paired_delta(values, nsplit):.6f}",
                        f"{(mean / statistics.mean(static_control) - 1.0) * 100.0:.6f}",
                        f"{paired_delta(values, static_control):.6f}",
                    )
                )

    lines = [
        "# N-split weight-stationary B-collector result",
        "",
        "Dense 16K BF16-input/FP32-output GEMM event TFLOP/s.  Each sample",
        "is one W1/I5 process; four Latin-rotated passes were collected.",
        "",
    ]
    for input_name in INPUTS:
        e7a = samples[(input_name, "e7a_exact")]
        nsplit = samples[(input_name, "nsplit_exact")]
        static_control = samples[(input_name, "nsplit_static_control")]
        lines.extend(
            (
                f"## Input `{input_name}`",
                "",
                "| variant | samples | mean +/- sample SD | vs E7a | "
                "vs N-split | paired vs N-split | paired vs static |",
                "|---|---|---:|---:|---:|---:|---:|",
            )
        )
        for variant in VARIANTS:
            values = samples[(input_name, variant)]
            mean = statistics.mean(values)
            formatted = ", ".join(f"{value:.3f}" for value in values)
            lines.append(
                f"| {variant} | {formatted} | "
                f"{mean:.3f} +/- {statistics.stdev(values):.3f} | "
                f"{(mean / statistics.mean(e7a) - 1.0) * 100.0:+.4f}% | "
                f"{(mean / statistics.mean(nsplit) - 1.0) * 100.0:+.4f}% | "
                f"{paired_delta(values, nsplit):+.4f}% | "
                f"{paired_delta(values, static_control):+.4f}% |"
            )
        lines.append("")

    lines.extend(
        (
            "## Decision rule",
            "",
            "- `nsplit_static_control` versus `nsplit_exact` isolates the",
            "  pipe-specialized consumer/code-shape effect.",
            "- Advance only if `nsplit_ws_b01` improves pass-matched",
            "  `nsplit_static_control` by at least 0.5% for both inputs.",
            "- Replace the performance reference only if it also beats exact",
            "  E7a in both distributions.",
            "",
        )
    )
    summary_path = result_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
