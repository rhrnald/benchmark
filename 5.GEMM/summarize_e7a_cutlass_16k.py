#!/usr/bin/env python3
"""Summarize the balanced 16K E7a/CUTLASS W1/I5 comparison."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


def read_e7a(result_dir: Path, pass_index: int) -> tuple[float, float]:
    path = result_dir / f"e7a_unit_16384_p{pass_index}.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one data row")
    return float(rows[0]["event_TFLOPS"]), float(rows[0]["event_ms"])


def read_cutlass(result_dir: Path, pass_index: int) -> tuple[float, float]:
    path = result_dir / "logs" / f"cutlass_p{pass_index}.log"
    text = path.read_text()
    runtime = re.findall(r"Avg runtime:\s+([0-9.eE+-]+)\s+ms", text)
    gflops = re.findall(r"GFLOPS:\s+([0-9.eE+-]+)", text)
    if len(runtime) != 1 or len(gflops) != 1:
        raise ValueError(f"{path}: expected one runtime and GFLOPS value")
    return float(gflops[0]) / 1000.0, float(runtime[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir
    values = {"e7a": [], "cutlass": []}
    runtimes = {"e7a": [], "cutlass": []}
    for pass_index in range(1, 5):
        for method, reader in (("e7a", read_e7a), ("cutlass", read_cutlass)):
            tflops, runtime = reader(result_dir, pass_index)
            values[method].append(tflops)
            runtimes[method].append(runtime)

    aggregate_path = result_dir / "aggregate.csv"
    with aggregate_path.open("w", newline="") as stream:
        fields = [
            "method",
            "processes",
            "run1_tflops",
            "run2_tflops",
            "run3_tflops",
            "run4_tflops",
            "mean_tflops",
            "sample_sd_tflops",
            "mean_runtime_ms",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method in ("e7a", "cutlass"):
            row = {
                "method": method,
                "processes": 4,
                **{
                    f"run{index}_tflops": f"{value:.3f}"
                    for index, value in enumerate(values[method], 1)
                },
                "mean_tflops": f"{statistics.mean(values[method]):.3f}",
                "sample_sd_tflops": f"{statistics.stdev(values[method]):.3f}",
                "mean_runtime_ms": f"{statistics.mean(runtimes[method]):.6f}",
            }
            writer.writerow(row)

    e7a_mean = statistics.mean(values["e7a"])
    cutlass_mean = statistics.mean(values["cutlass"])
    relative = (e7a_mean / cutlass_mean - 1.0) * 100.0
    table_path = result_dir / "comparison_table.md"
    rows = [
        "# 16K E7a vs CUTLASS TFLOP/s",
        "",
        "| implementation | process TFLOP/s | mean ± sample SD | mean runtime |",
        "|---|---:|---:|---:|",
        (
            "| clean E7a | "
            + " / ".join(f"{value:.3f}" for value in values["e7a"])
            + f" | **{e7a_mean:.3f} ± {statistics.stdev(values['e7a']):.3f}**"
            + f" | {statistics.mean(runtimes['e7a']):.6f} ms |"
        ),
        (
            "| CUTLASS selected 16K | "
            + " / ".join(f"{value:.3f}" for value in values["cutlass"])
            + f" | **{cutlass_mean:.3f} ± "
            + f"{statistics.stdev(values['cutlass']):.3f}**"
            + f" | {statistics.mean(runtimes['cutlass']):.6f} ms |"
        ),
        "",
        f"Clean E7a relative to CUTLASS: **{relative:+.3f}%**.",
        "",
    ]
    table_path.write_text("\n".join(rows))
    print(f"aggregate={aggregate_path} table={table_path}")


if __name__ == "__main__":
    main()
