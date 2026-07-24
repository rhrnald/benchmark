#!/usr/bin/env python3
"""Summarize the matched W1/I5 N-split redesign experiment."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


VARIANTS = (
    "e7a_exact",
    "nsplit_exact",
    "nsplit_u1_suspend_prod",
)
INPUTS = ("random", "random-signed8")
CONTROL = {
    "e7a_exact": "e7a_exact",
    "nsplit_exact": "nsplit_exact",
    "nsplit_u1_suspend_prod": "nsplit_exact",
}


def load_value(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one result row, got {len(rows)}")
    return float(rows[0]["event_TFLOPS"])


def fmt_samples(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)


def mean_paired_delta(values: list[float], control: list[float]) -> float:
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
                for pass_idx in range(1, 4)
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
                "mean_tflops",
                "sample_sd_tflops",
                "vs_e7a_ratio_of_means_percent",
                "vs_e7a_mean_paired_percent",
                "matched_control",
                "vs_control_ratio_of_means_percent",
                "vs_control_mean_paired_percent",
            )
        )
        for input_name in INPUTS:
            e7a = samples[(input_name, "e7a_exact")]
            e7a_mean = statistics.mean(e7a)
            for variant in VARIANTS:
                values = samples[(input_name, variant)]
                mean = statistics.mean(values)
                control_variant = CONTROL[variant]
                control = samples[(input_name, control_variant)]
                control_mean = statistics.mean(control)
                writer.writerow(
                    (
                        input_name,
                        variant,
                        *(f"{value:.6f}" for value in values),
                        f"{mean:.6f}",
                        f"{statistics.stdev(values):.6f}",
                        f"{(mean / e7a_mean - 1.0) * 100.0:.6f}",
                        f"{mean_paired_delta(values, e7a):.6f}",
                        control_variant,
                        f"{(mean / control_mean - 1.0) * 100.0:.6f}",
                        f"{mean_paired_delta(values, control):.6f}",
                    )
                )

    lines = [
        "# 256x256 N-split redesign result",
        "",
        "All values are event TFLOP/s for dense 16K BF16-input/FP32-output",
        "GEMM.  Each sample is one process with W1/I5; three position-rotated",
        "passes were collected per input and variant.",
        "",
    ]
    for input_name in INPUTS:
        e7a = samples[(input_name, "e7a_exact")]
        e7a_mean = statistics.mean(e7a)
        lines.extend(
            (
                f"## Input `{input_name}`",
                "",
                "| variant | samples | mean +/- sample SD | vs E7a | "
                "matched control | paired vs control |",
                "|---|---|---:|---:|---:|---:|",
            )
        )
        for variant in VARIANTS:
            values = samples[(input_name, variant)]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            control_variant = CONTROL[variant]
            control = samples[(input_name, control_variant)]
            control_mean = statistics.mean(control)
            lines.append(
                f"| {variant} | {fmt_samples(values)} | "
                f"{mean:.3f} +/- {sd:.3f} | "
                f"{(mean / e7a_mean - 1.0) * 100.0:+.4f}% | "
                f"{control_variant} "
                f"{(mean / control_mean - 1.0) * 100.0:+.4f}% | "
                f"{mean_paired_delta(values, control):+.4f}% |"
            )
        lines.append("")

    lines.extend(
        (
            "## Decision rule",
            "",
            "- Advance an N-split mechanism only when its pass-matched change",
            "  exceeds +0.5% for both input distributions.",
            "- Replacing the performance reference also requires beating exact",
            "  E7a on both distributions in this activation.",
            "- A smaller consistent gain is retained only as a diagnostic.",
            "",
        )
    )
    summary_path = result_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
