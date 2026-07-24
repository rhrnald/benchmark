#!/usr/bin/env python3
"""Summarize paired N-split transpose-compute and no-C-store controls."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


VARIANTS = (
    "nsplit_exact",
    "nsplit_transpose_scalar",
    "nsplit_exact_nostore",
    "nsplit_transpose_nostore",
)
INPUTS = ("random", "random-signed8")
PASSES = range(1, 5)
CONTROL = {
    "nsplit_exact": "nsplit_exact",
    "nsplit_transpose_scalar": "nsplit_exact",
    "nsplit_exact_nostore": "nsplit_exact_nostore",
    "nsplit_transpose_nostore": "nsplit_exact_nostore",
}


def load_row(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one result row, got {len(rows)}")
    return {
        "event_TFLOPS": float(rows[0]["event_TFLOPS"]),
        "event_ms": float(rows[0]["event_ms"]),
    }


def paired_percent(values: list[float], control: list[float]) -> list[float]:
    return [(value / base - 1.0) * 100.0 for value, base in zip(values, control)]


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    csv_dir = result_dir / "csv"

    rows: dict[tuple[str, str], list[dict[str, float]]] = {}
    for input_name in INPUTS:
        for variant in VARIANTS:
            rows[(input_name, variant)] = [
                load_row(csv_dir / f"{variant}_{input_name}_p{pass_idx}.csv")
                for pass_idx in PASSES
            ]

    aggregate_path = result_dir / "aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "input",
                "variant",
                "sample_1_tflops",
                "sample_2_tflops",
                "sample_3_tflops",
                "sample_4_tflops",
                "mean_tflops",
                "sample_sd_tflops",
                "mean_event_ms",
                "vs_mode_control_mean_paired_percent",
                "vs_mode_control_paired_sd_percent_points",
            )
        )
        for input_name in INPUTS:
            for variant in VARIANTS:
                variant_rows = rows[(input_name, variant)]
                values = [row["event_TFLOPS"] for row in variant_rows]
                times = [row["event_ms"] for row in variant_rows]
                base_values = [
                    row["event_TFLOPS"]
                    for row in rows[(input_name, CONTROL[variant])]
                ]
                delta = paired_percent(values, base_values)
                writer.writerow(
                    (
                        input_name,
                        variant,
                        *(f"{value:.6f}" for value in values),
                        f"{statistics.mean(values):.6f}",
                        f"{statistics.stdev(values):.6f}",
                        f"{statistics.mean(times):.9f}",
                        f"{statistics.mean(delta):.6f}",
                        f"{statistics.stdev(delta):.6f}",
                    )
                )

    lines = [
        "# N-split transpose-compute B200 result",
        "",
        "Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5",
        "process; four Latin-rotated passes were collected.",
        "",
    ]
    for input_name in INPUTS:
        lines.extend(
            (
                f"## Input `{input_name}`",
                "",
                "| variant | TFLOP/s samples | mean +/- sample SD | "
                "paired vs mode control | mean event ms |",
                "|---|---|---:|---:|---:|",
            )
        )
        for variant in VARIANTS:
            variant_rows = rows[(input_name, variant)]
            values = [row["event_TFLOPS"] for row in variant_rows]
            times = [row["event_ms"] for row in variant_rows]
            base_values = [
                row["event_TFLOPS"]
                for row in rows[(input_name, CONTROL[variant])]
            ]
            delta = paired_percent(values, base_values)
            lines.append(
                f"| {variant} | {', '.join(f'{value:.3f}' for value in values)} "
                f"| {statistics.mean(values):.3f} +/- "
                f"{statistics.stdev(values):.3f} | "
                f"{statistics.mean(delta):+.4f}% +/- "
                f"{statistics.stdev(delta):.4f}%p | "
                f"{statistics.mean(times):.6f} |"
            )

        exact_ms = [
            row["event_ms"] for row in rows[(input_name, "nsplit_exact")]
        ]
        transpose_ms = [
            row["event_ms"]
            for row in rows[(input_name, "nsplit_transpose_scalar")]
        ]
        exact_nostore_ms = [
            row["event_ms"]
            for row in rows[(input_name, "nsplit_exact_nostore")]
        ]
        transpose_nostore_ms = [
            row["event_ms"]
            for row in rows[(input_name, "nsplit_transpose_nostore")]
        ]
        exact_epilogue = [
            e2e - main
            for e2e, main in zip(exact_ms, exact_nostore_ms)
        ]
        transpose_epilogue = [
            e2e - main
            for e2e, main in zip(transpose_ms, transpose_nostore_ms)
        ]
        incremental = [
            transpose - exact
            for transpose, exact in zip(transpose_epilogue, exact_epilogue)
        ]
        exact_epi_mean, exact_epi_sd = mean_sd(exact_epilogue)
        transpose_epi_mean, transpose_epi_sd = mean_sd(transpose_epilogue)
        incremental_mean, incremental_sd = mean_sd(incremental)
        lines.extend(
            (
                "",
                "| derived per-launch time | mean +/- sample SD |",
                "|---|---:|",
                f"| exact epilogue (`E2E - no-store`) | "
                f"{exact_epi_mean:.6f} +/- {exact_epi_sd:.6f} ms |",
                f"| scalar-transpose epilogue (`E2E - no-store`) | "
                f"{transpose_epi_mean:.6f} +/- "
                f"{transpose_epi_sd:.6f} ms |",
                f"| incremental transpose epilogue | "
                f"{incremental_mean:+.6f} +/- "
                f"{incremental_sd:.6f} ms |",
                "",
            )
        )

    lines.extend(
        (
            "## Decision rule",
            "",
            "- Adopt only if `nsplit_transpose_scalar` improves pass-matched",
            "  `nsplit_exact` by at least 0.5% for both inputs.",
            "- If no-store improves but E2E does not, retain only the",
            "  transpose mapping as an epilogue-optimization candidate.",
            "",
        )
    )
    summary_path = result_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
