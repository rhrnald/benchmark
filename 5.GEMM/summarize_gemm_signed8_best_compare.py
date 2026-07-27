#!/usr/bin/env python3
import csv
import re
import statistics
import sys
from pathlib import Path


def parse_ours(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one data row in {path}, got {len(rows)}")
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
    sizes = (8192, 16384, 32768)
    methods = ("cutlass", "cublas", "ours")
    values: dict[tuple[int, str], list[float]] = {}

    for size in sizes:
        for method in methods:
            samples = []
            for run in (1, 2, 3):
                if method == "ours":
                    samples.append(
                        parse_ours(out_dir / "csv" / f"ours_{size}_p{run}.csv")
                    )
                else:
                    samples.append(
                        parse_library(
                            out_dir / "logs" / f"{method}_{size}_p{run}.log",
                            method,
                        )
                    )
            values[size, method] = samples

    aggregate = out_dir / "aggregate.csv"
    with aggregate.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["size", "method", "sample1", "sample2", "sample3", "mean", "sample_sd"]
        )
        for size in sizes:
            for method in methods:
                samples = values[size, method]
                writer.writerow(
                    [
                        size,
                        method,
                        *[f"{x:.3f}" for x in samples],
                        f"{statistics.mean(samples):.3f}",
                        f"{statistics.stdev(samples):.3f}",
                    ]
                )

    lines = [
        "# BF16 uniform `[-8,8)` GEMM comparison",
        "",
        "Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.",
        "Each cell is mean +/- sample SD across three independent processes;",
        "each process used one warmup and five timed launches.",
        "",
        "| size | CUTLASS | cuBLAS | ours (best candidate) | ours vs cuBLAS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size in sizes:
        means = {m: statistics.mean(values[size, m]) for m in methods}
        sds = {m: statistics.stdev(values[size, m]) for m in methods}
        delta = 100.0 * (means["ours"] / means["cublas"] - 1.0)
        lines.append(
            f"| {size // 1024}K | "
            f"{means['cutlass']:.3f} +/- {sds['cutlass']:.3f} | "
            f"{means['cublas']:.3f} +/- {sds['cublas']:.3f} | "
            f"{means['ours']:.3f} +/- {sds['ours']:.3f} | "
            f"{delta:+.3f}% |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
