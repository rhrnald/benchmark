#!/usr/bin/env python3
import csv
import re
import statistics
import sys
from pathlib import Path


INPUTS = ("unit", "signed8")
METHODS = ("ours", "cublas", "cutlass")
PASSES = range(1, 7)


def parse_ours(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    return float(rows[0]["event_TFLOPS"])


def parse_library(path: Path, method: str) -> float:
    text = path.read_text()
    pattern = (
        r"event_TFLOPS=([0-9.]+)"
        if method == "cublas"
        else r"GFLOPS:\s*([0-9.eE+-]+)"
    )
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing throughput in {path}")
    value = float(match.group(1))
    return value if method == "cublas" else value / 1000.0


def main() -> None:
    out_dir = Path(sys.argv[1])
    values: dict[tuple[str, str], list[float]] = {}
    for input_name in INPUTS:
        for method in METHODS:
            samples = []
            for run in PASSES:
                if method == "ours":
                    samples.append(
                        parse_ours(
                            out_dir / "csv" / f"{input_name}_{method}_p{run}.csv"
                        )
                    )
                else:
                    samples.append(
                        parse_library(
                            out_dir
                            / "logs"
                            / f"{input_name}_{method}_p{run}.log",
                            method,
                        )
                    )
            values[input_name, method] = samples

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input",
                "method",
                *[f"sample{i}" for i in PASSES],
                "mean",
                "sample_sd",
                "relative_to_cublas_pct",
            ]
        )
        for input_name in INPUTS:
            cublas_mean = statistics.mean(values[input_name, "cublas"])
            for method in METHODS:
                samples = values[input_name, method]
                mean = statistics.mean(samples)
                writer.writerow(
                    [
                        input_name,
                        method,
                        *[f"{x:.3f}" for x in samples],
                        f"{mean:.3f}",
                        f"{statistics.stdev(samples):.3f}",
                        f"{100.0 * mean / cublas_mean:.3f}",
                    ]
                )

    labels = {"unit": "`[0,1)`", "signed8": "`[-8,8)`"}
    lines = [
        "# 16K BF16 GEMM: ours vs cuBLAS vs CUTLASS",
        "",
        "Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.",
        "Each cell is mean +/- sample SD across six independent processes.",
        "Each process uses one warmup and five timed launches. The six method",
        "orders are fully position-balanced.",
        "",
        "| input | ours (`fixed_sink`) | cuBLAS | CUTLASS selected CLC | ours/cuBLAS |",
        "|---|---:|---:|---:|---:|",
    ]
    for input_name in INPUTS:
        means = {
            method: statistics.mean(values[input_name, method])
            for method in METHODS
        }
        sds = {
            method: statistics.stdev(values[input_name, method])
            for method in METHODS
        }
        lines.append(
            f"| {labels[input_name]} | "
            f"{means['ours']:.3f} +/- {sds['ours']:.3f} | "
            f"{means['cublas']:.3f} +/- {sds['cublas']:.3f} | "
            f"{means['cutlass']:.3f} +/- {sds['cutlass']:.3f} | "
            f"{100.0 * means['ours'] / means['cublas']:.3f}% |"
        )
    lines.append("")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
