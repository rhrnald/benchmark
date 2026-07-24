#!/usr/bin/env python3
"""Summarize the matched scalar x64-versus-x32 TMEM-load ablation."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from collections import Counter
from pathlib import Path


VARIANTS = (
    "nsplit_exact",
    "nsplit_transpose_scalar_x64",
    "nsplit_transpose_scalar_x32",
)
INPUTS = ("random", "random-signed8")
PASSES = range(1, 7)
T95_DF5 = 2.5705818366


def validate_sequence(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        sequence_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(sequence_rows) != len(INPUTS) * len(PASSES):
        raise RuntimeError(
            f"{path}: expected 12 sequence rows, got {len(sequence_rows)}"
        )

    expected_permutations = set(itertools.permutations(VARIANTS))
    by_key: dict[tuple[str, int], tuple[str, ...]] = {}
    for row in sequence_rows:
        input_name = row.get("input", "")
        if input_name not in INPUTS:
            raise RuntimeError(f"{path}: unexpected input {input_name!r}")
        try:
            pass_idx = int(row.get("pass", ""))
        except ValueError as exc:
            raise RuntimeError(f"{path}: invalid pass {row.get('pass')!r}") from exc
        order = tuple(row.get("order", "").split())
        key = (input_name, pass_idx)
        if key in by_key:
            raise RuntimeError(f"{path}: duplicate sequence row {key}")
        if pass_idx not in PASSES:
            raise RuntimeError(f"{path}: pass out of range: {pass_idx}")
        if len(order) != len(VARIANTS) or set(order) != set(VARIANTS):
            raise RuntimeError(f"{path}: invalid order for {key}: {order}")
        by_key[key] = order

    for input_name in INPUTS:
        orders = [by_key[(input_name, pass_idx)] for pass_idx in PASSES]
        if set(orders) != expected_permutations:
            raise RuntimeError(
                f"{path}: {input_name} does not contain all six permutations"
            )
        positions = Counter(
            (variant, position)
            for order in orders
            for position, variant in enumerate(order)
        )
        if any(
            positions[(variant, position)] != 2
            for variant in VARIANTS
            for position in range(len(VARIANTS))
        ):
            raise RuntimeError(f"{path}: {input_name} position imbalance")
        adjacencies = Counter(
            pair
            for order in orders
            for pair in zip(order, order[1:])
        )
        if any(
            adjacencies[(left, right)] != 2
            for left in VARIANTS
            for right in VARIANTS
            if left != right
        ):
            raise RuntimeError(f"{path}: {input_name} adjacency imbalance")


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
        "mtile": "64",
        "ntile": "64",
        "ktiles": "256",
        "ctas": "4096",
        "launch_ctas": "148",
        "warmup": "1",
        "iters": "5",
        "input_init": expected_input,
        "dynamic_smem_bytes": "197632",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise RuntimeError(
                f"{path}: expected {field}={value}, got {row.get(field)!r}"
            )
    if "B200" not in row.get("device", ""):
        raise RuntimeError(f"{path}: expected a B200 device")

    event_tflops = float(row["event_TFLOPS"])
    event_ms = float(row["event_ms"])
    if not math.isfinite(event_tflops) or event_tflops <= 0.0:
        raise RuntimeError(f"{path}: invalid event_TFLOPS={event_tflops}")
    if not math.isfinite(event_ms) or event_ms <= 0.0:
        raise RuntimeError(f"{path}: invalid event_ms={event_ms}")
    return {"event_TFLOPS": event_tflops, "event_ms": event_ms}


def paired_percent(values: list[float], control: list[float]) -> list[float]:
    if len(values) != len(control):
        raise RuntimeError("paired vectors have different lengths")
    return [(value / base - 1.0) * 100.0 for value, base in zip(values, control)]


def mean_sd_ci(
    values: list[float],
) -> tuple[float, float, float, float]:
    if len(values) != len(PASSES):
        raise RuntimeError(f"expected six samples, got {len(values)}")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95_DF5 * sd / math.sqrt(len(values))
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
    validate_sequence(result_dir / "sequence.tsv")

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
                "vs_x64_mean_paired_percent",
                "vs_x64_ci95_low_percent",
                "vs_x64_ci95_high_percent",
                "vs_exact_mean_paired_percent",
                "vs_exact_ci95_low_percent",
                "vs_exact_ci95_high_percent",
            )
        )
        for input_name in INPUTS:
            x64 = [
                row["event_TFLOPS"]
                for row in rows[
                    (input_name, "nsplit_transpose_scalar_x64")
                ]
            ]
            exact = [
                row["event_TFLOPS"]
                for row in rows[(input_name, "nsplit_exact")]
            ]
            for variant in VARIANTS:
                variant_rows = rows[(input_name, variant)]
                values = [row["event_TFLOPS"] for row in variant_rows]
                times = [row["event_ms"] for row in variant_rows]
                x64_stats = mean_sd_ci(paired_percent(values, x64))
                exact_stats = mean_sd_ci(paired_percent(values, exact))
                writer.writerow(
                    (
                        input_name,
                        variant,
                        *(f"{value:.6f}" for value in values),
                        f"{statistics.mean(values):.6f}",
                        f"{statistics.stdev(values):.6f}",
                        f"{statistics.mean(times):.9f}",
                        f"{x64_stats[0]:.6f}",
                        f"{x64_stats[2]:.6f}",
                        f"{x64_stats[3]:.6f}",
                        f"{exact_stats[0]:.6f}",
                        f"{exact_stats[2]:.6f}",
                        f"{exact_stats[3]:.6f}",
                    )
                )

    lines = [
        "# N-split scalar x32 TMEM-load B200 result",
        "",
        "Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5",
        "process. All six orders of the three variants were collected per",
        "input; each variant occupied each position twice and every ordered",
        "non-self adjacent pair occurred twice within those orders.",
        "",
    ]
    candidate_effects: dict[str, tuple[float, float, float]] = {}
    for input_name in INPUTS:
        x64 = [
            row["event_TFLOPS"]
            for row in rows[(input_name, "nsplit_transpose_scalar_x64")]
        ]
        exact = [
            row["event_TFLOPS"]
            for row in rows[(input_name, "nsplit_exact")]
        ]
        lines.extend(
            (
                f"## Input `{input_name}`",
                "",
                "| variant | mean +/- sample SD TFLOP/s | vs scalar x64, "
                "paired 95% CI | vs exact, paired 95% CI | mean event ms |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for variant in VARIANTS:
            variant_rows = rows[(input_name, variant)]
            values = [row["event_TFLOPS"] for row in variant_rows]
            times = [row["event_ms"] for row in variant_rows]
            x64_delta = paired_percent(values, x64)
            exact_delta = paired_percent(values, exact)
            lines.append(
                f"| {variant} | {statistics.mean(values):.3f} +/- "
                f"{statistics.stdev(values):.3f} | "
                f"{fmt_effect(x64_delta)} | {fmt_effect(exact_delta)} | "
                f"{statistics.mean(times):.6f} |"
            )
            if variant == "nsplit_transpose_scalar_x32":
                stats = mean_sd_ci(x64_delta)
                candidate_effects[input_name] = (
                    stats[0],
                    stats[2],
                    stats[3],
                )

        x64_times = [
            row["event_ms"]
            for row in rows[
                (input_name, "nsplit_transpose_scalar_x64")
            ]
        ]
        x32_times = [
            row["event_ms"]
            for row in rows[
                (input_name, "nsplit_transpose_scalar_x32")
            ]
        ]
        time_delta = [
            x32 - x64 for x32, x64 in zip(x32_times, x64_times)
        ]
        mean, sd, low, high = mean_sd_ci(time_delta)
        lines.extend(
            (
                "",
                "Scalar x32 minus scalar x64 event time:",
                f"`{mean:+.6f} +/- {sd:.6f} ms`, paired 95% CI "
                f"`[{low:+.6f},{high:+.6f}] ms`.",
                "",
            )
        )

    advances = all(candidate_effects[name][1] > 0.0 for name in INPUTS)
    lines.extend(
        (
            "## Decision",
            "",
            "The scalar x32 candidate "
            + ("passes" if advances else "does not pass")
            + " the screening gate. The gate requires the paired 95% CI",
            "against scalar x64 to be above zero for both input",
            "distributions.",
            "",
            "A passing sweep candidate still requires a separately committed",
            "counterbalanced x64/x32 confirmation before adoption. Canonical",
            "replacement additionally requires at least +0.5% over exact for",
            "both inputs with no negative paired 95% interval.",
            "",
        )
    )

    summary_path = result_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
