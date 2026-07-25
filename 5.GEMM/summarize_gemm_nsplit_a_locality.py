#!/usr/bin/env python3
"""Summarize clean N-split A-locality scheduler sweep CSVs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


VARIANTS = ("baseline", "nfast_16x16", "nfast_8x32", "nfast_4x64")
INPUTS = ("random", "random-signed8")


def load_value(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one row, got {len(rows)}")
    row = rows[0]
    expected = {
        "size": "16384",
        "cta_m": "256",
        "cta_n": "256",
        "stage_k": "64",
        "stages": "3",
        "warmup": "1",
        "iters": "5",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"{path}: expected {key}={value}, got {row.get(key)!r}")
    return float(row["event_TFLOPS"])


def samples_for(csv_dir: Path, input_name: str, variant: str) -> list[float]:
    return [
        load_value(csv_dir / f"{variant}_{input_name}_p{pass_idx}.csv")
        for pass_idx in range(1, 5)
    ]


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
            samples[(input_name, variant)] = samples_for(csv_dir, input_name, variant)

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
                "ratio_of_means_delta_percent",
                "mean_paired_delta_percent",
            )
        )
        for input_name in INPUTS:
            baseline = samples[(input_name, "baseline")]
            baseline_mean = statistics.mean(baseline)
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
                        f"{(mean / baseline_mean - 1.0) * 100.0:.6f}",
                        f"{statistics.mean((v / b - 1.0) * 100.0 for v, b in zip(values, baseline)):.6f}",
                    )
                )

    lines = [
        "# Clean N-split A-locality scheduler sweep",
        "",
        "Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5",
        "process. All variants keep the same mainloop, TMA traffic, C store,",
        "and 148 persistent CTAs; only output-tile order changes.",
        "",
    ]
    for input_name in INPUTS:
        baseline = samples[(input_name, "baseline")]
        baseline_mean = statistics.mean(baseline)
        lines.extend(
            [
                f"## Input `{input_name}`",
                "",
                "| variant | samples | mean +/- SD | vs baseline | paired vs baseline |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            values = samples[(input_name, variant)]
            mean = statistics.mean(values)
            paired = statistics.mean(
                (v / b - 1.0) * 100.0 for v, b in zip(values, baseline)
            )
            lines.append(
                f"| `{variant}` | {fmt_samples(values)} | "
                f"{mean:.3f} +/- {statistics.stdev(values):.3f} | "
                f"{(mean / baseline_mean - 1.0) * 100.0:+.4f}% | "
                f"{paired:+.4f}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate",
            "",
            "Promote only if both input distributions improve by at least 0.5%",
            "against baseline and no validation or resource gate regresses.",
            "",
        ]
    )
    summary_path = result_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
