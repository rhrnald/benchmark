#!/usr/bin/env python3
"""Summarize the matched N-split transpose-epilogue ablation."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


VARIANTS = (
    "nsplit_exact",
    "nsplit_transpose_nostore",
    "nsplit_transpose_scalar",
    "nsplit_transpose_vec2_x32",
    "nsplit_transpose_vec2_cf1_x32",
    "nsplit_transpose_vec2_cf_x32",
    "nsplit_transpose_vec2_cf_x64",
    "nsplit_transpose_vec4_cf_x64",
)
EPILOGUE_VARIANTS = VARIANTS[2:]
INPUTS = ("random", "random-signed8")
PASSES = range(1, 9)
T95_DF7 = 2.364624251


def load_row(path: Path, expected_input: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one result row, got {len(rows)}")
    row = rows[0]
    expected = {
        "size": "16384",
        "m": "16384",
        "n": "16384",
        "k": "16384",
        "cta_m": "256",
        "cta_n": "256",
        "stage_k": "64",
        "stages": "3",
        "launch_ctas": "148",
        "warmup": "1",
        "iters": "5",
        "input_init": expected_input,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise RuntimeError(
                f"{path}: expected {field}={value}, got {row.get(field)!r}"
            )
    if "B200" not in row.get("device", ""):
        raise RuntimeError(f"{path}: expected a B200 device")
    return {
        "event_TFLOPS": float(row["event_TFLOPS"]),
        "event_ms": float(row["event_ms"]),
    }


def paired_percent(values: list[float], control: list[float]) -> list[float]:
    return [(value / base - 1.0) * 100.0 for value, base in zip(values, control)]


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95_DF7 * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def fmt_effect(values: list[float]) -> str:
    mean, _, low, high = mean_sd_ci(values)
    return f"{mean:+.4f}% [{low:+.4f},{high:+.4f}]"


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
                load_row(
                    csv_dir / f"{variant}_{input_name}_p{pass_idx}.csv",
                    input_name,
                )
                for pass_idx in PASSES
            ]

    aggregate_path = result_dir / "aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "input",
                "variant",
                *(f"sample_{i}_tflops" for i in PASSES),
                "mean_tflops",
                "sample_sd_tflops",
                "mean_event_ms",
                "vs_scalar_mean_paired_percent",
                "vs_scalar_ci95_low_percent",
                "vs_scalar_ci95_high_percent",
                "vs_exact_mean_paired_percent",
                "vs_exact_ci95_low_percent",
                "vs_exact_ci95_high_percent",
            )
        )
        for input_name in INPUTS:
            scalar = [
                row["event_TFLOPS"]
                for row in rows[(input_name, "nsplit_transpose_scalar")]
            ]
            exact = [
                row["event_TFLOPS"]
                for row in rows[(input_name, "nsplit_exact")]
            ]
            for variant in VARIANTS:
                variant_rows = rows[(input_name, variant)]
                values = [row["event_TFLOPS"] for row in variant_rows]
                times = [row["event_ms"] for row in variant_rows]
                scalar_delta = paired_percent(values, scalar)
                exact_delta = paired_percent(values, exact)
                scalar_stats = mean_sd_ci(scalar_delta)
                exact_stats = mean_sd_ci(exact_delta)
                writer.writerow(
                    (
                        input_name,
                        variant,
                        *(f"{value:.6f}" for value in values),
                        f"{statistics.mean(values):.6f}",
                        f"{statistics.stdev(values):.6f}",
                        f"{statistics.mean(times):.9f}",
                        f"{scalar_stats[0]:.6f}",
                        f"{scalar_stats[2]:.6f}",
                        f"{scalar_stats[3]:.6f}",
                        f"{exact_stats[0]:.6f}",
                        f"{exact_stats[2]:.6f}",
                        f"{exact_stats[3]:.6f}",
                    )
                )

    lines = [
        "# N-split transpose-epilogue B200 result",
        "",
        "Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5",
        "process. Eight Williams-balanced orders were collected per input,",
        "so every variant occupied every execution position exactly once and",
        "within those orders every ordered non-self adjacent pair occurred",
        "exactly once.",
        "",
    ]

    joint_effects: dict[str, list[float]] = {
        variant: [] for variant in EPILOGUE_VARIANTS
    }
    for input_name in INPUTS:
        scalar = [
            row["event_TFLOPS"]
            for row in rows[(input_name, "nsplit_transpose_scalar")]
        ]
        exact = [
            row["event_TFLOPS"]
            for row in rows[(input_name, "nsplit_exact")]
        ]
        nostore_ms = [
            row["event_ms"]
            for row in rows[(input_name, "nsplit_transpose_nostore")]
        ]
        lines.extend(
            (
                f"## Input `{input_name}`",
                "",
                "| variant | mean +/- sample SD TFLOP/s | vs scalar, paired "
                "95% CI | vs exact, paired 95% CI | mean event ms |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for variant in VARIANTS:
            variant_rows = rows[(input_name, variant)]
            values = [row["event_TFLOPS"] for row in variant_rows]
            times = [row["event_ms"] for row in variant_rows]
            scalar_delta = paired_percent(values, scalar)
            exact_delta = paired_percent(values, exact)
            if variant in joint_effects:
                joint_effects[variant].append(statistics.mean(scalar_delta))
            lines.append(
                f"| {variant} | {statistics.mean(values):.3f} +/- "
                f"{statistics.stdev(values):.3f} | "
                f"{fmt_effect(scalar_delta)} | {fmt_effect(exact_delta)} | "
                f"{statistics.mean(times):.6f} |"
            )

        lines.extend(
            (
                "",
                "| transpose epilogue | E2E - transpose no-store, "
                "mean +/- sample SD ms |",
                "|---|---:|",
            )
        )
        for variant in EPILOGUE_VARIANTS:
            e2e_ms = [
                row["event_ms"] for row in rows[(input_name, variant)]
            ]
            epilogue_ms = [
                e2e - main for e2e, main in zip(e2e_ms, nostore_ms)
            ]
            lines.append(
                f"| {variant} | {statistics.mean(epilogue_ms):.6f} +/- "
                f"{statistics.stdev(epilogue_ms):.6f} |"
            )
        lines.append("")

    ranked = sorted(
        EPILOGUE_VARIANTS,
        key=lambda variant: min(joint_effects[variant]),
        reverse=True,
    )
    lines.extend(
        (
            "## Cross-input ranking",
            "",
            "Ranking uses the smaller of the two mean paired improvements",
            "against the scalar-transpose control.",
            "",
            "| rank | variant | `[0,1)` | `[-8,8)` | worst input |",
            "|---:|---|---:|---:|---:|",
        )
    )
    for rank, variant in enumerate(ranked, start=1):
        random_effect, signed_effect = joint_effects[variant]
        lines.append(
            f"| {rank} | {variant} | {random_effect:+.4f}% | "
            f"{signed_effect:+.4f}% | "
            f"{min(random_effect, signed_effect):+.4f}% |"
        )

    lines.extend(
        (
            "",
            "## Decision rule",
            "",
            "- Correctness, zero spill/local memory, and the expected MMA/TMA",
            "  instruction counts are mandatory.",
            "- An epilogue candidate advances only when its paired 95% CI",
            "  against `nsplit_transpose_scalar` is above zero for both inputs.",
            "- The sweep winner requires a separate confirmatory matched A/B",
            "  run before adoption. It replaces `nsplit_exact` only when that",
            "  run shows at least +0.5% over exact for both inputs without a",
            "  negative paired 95% CI.",
            "",
        )
    )

    summary_path = result_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
