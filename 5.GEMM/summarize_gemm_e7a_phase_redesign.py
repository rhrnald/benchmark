#!/usr/bin/env python3
"""Summarize W1/I5 E7a phase-redesign CSVs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


VARIANTS = (
    "baseline",
    "cta4",
    "cta8",
    "b1_gap0",
    "b1_gap32",
    "b1_gap64",
    "b1_cross",
)
INPUTS = ("random", "random-signed8")


def load_value(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one result row, got {len(rows)}")
    return float(rows[0]["event_TFLOPS"])


def fmt_samples(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    csv_dir = result_dir / "csv"

    samples: dict[tuple[str, str], list[float]] = {}
    for input_name in INPUTS:
        for variant in VARIANTS:
            values = [
                load_value(csv_dir / f"{variant}_{input_name}_p{pass_idx}.csv")
                for pass_idx in range(1, 4)
            ]
            samples[(input_name, variant)] = values

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
                "ratio_of_means_delta_percent",
                "mean_paired_delta_percent",
                "matched_control",
                "matched_control_delta_percent",
                "mean_paired_control_delta_percent",
            )
        )
        for input_name in INPUTS:
            baseline = samples[(input_name, "baseline")]
            baseline_mean = statistics.mean(baseline)
            for variant in VARIANTS:
                values = samples[(input_name, variant)]
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                ratio_delta = (mean / baseline_mean - 1.0) * 100.0
                paired_delta = statistics.mean(
                    (value / base - 1.0) * 100.0
                    for value, base in zip(values, baseline)
                )
                control_variant = (
                    "b1_gap0"
                    if variant in {"b1_gap32", "b1_gap64"}
                    else "baseline"
                )
                control = samples[(input_name, control_variant)]
                control_mean = statistics.mean(control)
                control_delta = (mean / control_mean - 1.0) * 100.0
                paired_control_delta = statistics.mean(
                    (value / base - 1.0) * 100.0
                    for value, base in zip(values, control)
                )
                writer.writerow(
                    (
                        input_name,
                        variant,
                        *(f"{value:.6f}" for value in values),
                        f"{mean:.6f}",
                        f"{sd:.6f}",
                        f"{ratio_delta:.6f}",
                        f"{paired_delta:.6f}",
                        control_variant,
                        f"{control_delta:.6f}",
                        f"{paired_control_delta:.6f}",
                    )
                )

    lines = [
        "# E7a phase-redesign ablation",
        "",
        "All results are 16K dense BF16-input/FP32-output GEMM event TFLOP/s.",
        "Each sample is a separate process with one warmup and five timed",
        "launches.  Three rotated passes were collected for each input.",
        "",
        "The clean E7a baseline is compiled directly from the audited canonical",
        "source. Candidate sources are hash-gated generated variants.",
        "",
    ]
    for input_name in INPUTS:
        baseline = samples[(input_name, "baseline")]
        baseline_mean = statistics.mean(baseline)
        lines.extend(
            (
                f"## Input `{input_name}`",
                "",
                "| variant | samples | mean +/- sample SD | vs baseline | "
                "vs matched control | paired control delta |",
                "|---|---|---:|---:|---:|---:|",
            )
        )
        for variant in VARIANTS:
            values = samples[(input_name, variant)]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            ratio_delta = (mean / baseline_mean - 1.0) * 100.0
            paired_delta = statistics.mean(
                (value / base - 1.0) * 100.0
                for value, base in zip(values, baseline)
            )
            control_variant = (
                "b1_gap0"
                if variant in {"b1_gap32", "b1_gap64"}
                else "baseline"
            )
            control = samples[(input_name, control_variant)]
            control_mean = statistics.mean(control)
            control_delta = (mean / control_mean - 1.0) * 100.0
            paired_control_delta = statistics.mean(
                (value / base - 1.0) * 100.0
                for value, base in zip(values, control)
            )
            lines.append(
                f"| {variant} | {fmt_samples(values)} | "
                f"{mean:.3f} +/- {sd:.3f} | {ratio_delta:+.4f}% | "
                f"{control_variant} {control_delta:+.4f}% | "
                f"{paired_control_delta:+.4f}% |"
            )
        lines.append("")

    lines.extend(
        (
            "## Interpretation gate",
            "",
            "- Adopt only a mechanism that improves both distributions by at",
            "  least 0.5% in paired measurements.",
            "- A consistent but smaller gain is diagnostic and requires a",
            "  pre-registered extension before any adoption claim.",
            "- `cta4/cta8` change only the one-time CTA startup phase;",
            "  `b1_gap0` controls for its code-generation change;",
            "  `b1_gap32/64` change only late-B1 spacing per K64 stage;",
            "  `b1_cross` relocates one existing W3 wait without adding work.",
            "",
        )
    )
    (result_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"wrote {aggregate_path}")
    print(f"wrote {result_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
