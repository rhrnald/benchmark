#!/usr/bin/env python3
import csv
import re
import statistics
import sys
from pathlib import Path


SIZES = (8192, 16384, 32768)
INPUTS = ("unit", "signed8")
METHODS = ("ours", "cublas")
PASSES = (1, 2, 3, 4)


def parse_ours(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    return float(rows[0]["event_TFLOPS"])


def parse_cublas(path: Path) -> float:
    match = re.search(r"event_TFLOPS=([0-9.]+)", path.read_text())
    if not match:
        raise RuntimeError(f"missing event_TFLOPS in {path}")
    return float(match.group(1))


def main() -> None:
    out_dir = Path(sys.argv[1])
    values: dict[tuple[int, str, str], list[float]] = {}

    for size in SIZES:
        for input_name in INPUTS:
            for method in METHODS:
                samples = []
                for run in PASSES:
                    if method == "ours":
                        path = (
                            out_dir
                            / "csv"
                            / f"{size}_{input_name}_{method}_p{run}.csv"
                        )
                        samples.append(parse_ours(path))
                    else:
                        path = (
                            out_dir
                            / "logs"
                            / f"{size}_{input_name}_{method}_p{run}.log"
                        )
                        samples.append(parse_cublas(path))
                values[size, input_name, method] = samples

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "size",
                "input",
                "method",
                *[f"sample{i}" for i in PASSES],
                "mean",
                "sample_sd",
                "ours_over_cublas_pct",
            ]
        )
        for size in SIZES:
            for input_name in INPUTS:
                ours_mean = statistics.mean(values[size, input_name, "ours"])
                cublas_mean = statistics.mean(values[size, input_name, "cublas"])
                ratio = 100.0 * ours_mean / cublas_mean
                for method in METHODS:
                    samples = values[size, input_name, method]
                    writer.writerow(
                        [
                            size,
                            input_name,
                            method,
                            *[f"{x:.3f}" for x in samples],
                            f"{statistics.mean(samples):.3f}",
                            f"{statistics.stdev(samples):.3f}",
                            f"{ratio:.3f}",
                        ]
                    )

    labels = {"unit": "`[0,1)`", "signed8": "`[-8,8)`"}
    lines = [
        "# BF16 GEMM ours vs cuBLAS remeasurement",
        "",
        "Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.",
        "Each cell is mean +/- sample SD across four independent processes;",
        "each process uses one warmup and five timed launches. The two methods",
        "occupy the first and second positions exactly twice per cell.",
        "",
        "| size | input | ours (static `8x16`) | cuBLAS | ours/cuBLAS |",
        "|---:|---|---:|---:|---:|",
    ]
    for size in SIZES:
        for input_name in INPUTS:
            ours = values[size, input_name, "ours"]
            cublas = values[size, input_name, "cublas"]
            ours_mean = statistics.mean(ours)
            cublas_mean = statistics.mean(cublas)
            lines.append(
                f"| {size // 1024}K | {labels[input_name]} | "
                f"{ours_mean:.3f} +/- {statistics.stdev(ours):.3f} | "
                f"{cublas_mean:.3f} +/- {statistics.stdev(cublas):.3f} | "
                f"{100.0 * ours_mean / cublas_mean:.3f}% |"
            )
    lines.append("")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
