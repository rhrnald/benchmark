#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


INPUTS = ("random", "random-signed8")
METHODS = ("gemm", "tma_only")
PASSES = range(1, 5)
SIZE = 16384
CTA_TILES = (SIZE // 256) ** 2
K_TILES = SIZE // 64
BYTES_PER_K_TILE = 256 * 64 * 2 + 2 * 64 * 128 * 2
LOGICAL_BYTES = CTA_TILES * K_TILES * BYTES_PER_K_TILE


def read_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def main() -> None:
    out_dir = Path(sys.argv[1])
    times: dict[tuple[str, str], list[float]] = {}
    tflops: dict[str, list[float]] = {}
    for input_name in INPUTS:
        for method in METHODS:
            rows = [
                read_row(
                    out_dir / "csv" / f"{input_name}_{method}_p{run}.csv"
                )
                for run in PASSES
            ]
            times[input_name, method] = [
                float(row["event_ms"]) for row in rows
            ]
            if method == "gemm":
                tflops[input_name] = [
                    float(row["event_TFLOPS"]) for row in rows
                ]

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input",
                "method",
                "logical_bytes",
                "sample1_ms",
                "sample2_ms",
                "sample3_ms",
                "sample4_ms",
                "mean_ms",
                "sample_sd_ms",
                "logical_TBps",
            ]
        )
        for input_name in INPUTS:
            for method in METHODS:
                samples = times[input_name, method]
                mean_ms = statistics.mean(samples)
                writer.writerow(
                    [
                        input_name,
                        method,
                        LOGICAL_BYTES,
                        *[f"{x:.6f}" for x in samples],
                        f"{mean_ms:.6f}",
                        f"{statistics.stdev(samples):.6f}",
                        f"{LOGICAL_BYTES / (mean_ms * 1e-3) / 1e12:.3f}",
                    ]
                )

    labels = {"random": "`[0,1)`", "random-signed8": "`[-8,8)`"}
    lines = [
        "# 16K dense TMA-load-only comparison",
        "",
        f"Logical A/B request bytes per launch: {LOGICAL_BYTES} B (64 GiB).",
        "Mean +/- sample SD across four independent W1/I5 processes.",
        "",
        "| input | GEMM TFLOP/s | GEMM ms | TMA-only ms | TMA-only logical TB/s | TMA-only/GEMM time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for input_name in INPUTS:
        gemm_ms = statistics.mean(times[input_name, "gemm"])
        load_ms = statistics.mean(times[input_name, "tma_only"])
        lines.append(
            f"| {labels[input_name]} | "
            f"{statistics.mean(tflops[input_name]):.3f} +/- "
            f"{statistics.stdev(tflops[input_name]):.3f} | "
            f"{gemm_ms:.6f} +/- "
            f"{statistics.stdev(times[input_name, 'gemm']):.6f} | "
            f"{load_ms:.6f} +/- "
            f"{statistics.stdev(times[input_name, 'tma_only']):.6f} | "
            f"{LOGICAL_BYTES / (load_ms * 1e-3) / 1e12:.3f} | "
            f"{100.0 * load_ms / gemm_ms:.2f}% |"
        )
    lines.append("")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
