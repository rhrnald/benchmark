#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


ORDERS = ("mfast", "nfast")
INPUTS = ("random", "random-signed8")
PASSES = (1, 2, 3, 4)


def read_tflops(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    return float(rows[0]["event_TFLOPS"])


def main() -> None:
    out_dir = Path(sys.argv[1])
    values = {
        (input_name, order): [
            read_tflops(
                out_dir / "csv" / f"{input_name}_{order}_p{run}.csv"
            )
            for run in PASSES
        ]
        for input_name in INPUTS
        for order in ORDERS
    }

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input",
                "order",
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "mean",
                "sample_sd",
                "nfast_vs_mfast_pct",
            ]
        )
        for input_name in INPUTS:
            baseline = statistics.mean(values[input_name, "mfast"])
            for order in ORDERS:
                samples = values[input_name, order]
                mean = statistics.mean(samples)
                writer.writerow(
                    [
                        input_name,
                        order,
                        *[f"{value:.3f}" for value in samples],
                        f"{mean:.3f}",
                        f"{statistics.stdev(samples):.3f}",
                        f"{100.0 * (mean / baseline - 1.0):+.4f}",
                    ]
                )

    lines = [
        "# Static 8x16 local-order comparison",
        "",
        "16K square BF16 GEMM, FP32 accumulation/output, 148 persistent CTAs,",
        "static grid-stride ownership, W1/I5, four ABBA-position-balanced",
        "independent process samples per cell.",
        "",
        "| input | M-fast | N-fast | N-fast vs M-fast |",
        "|---|---:|---:|---:|",
    ]
    for input_name in INPUTS:
        mfast = values[input_name, "mfast"]
        nfast = values[input_name, "nfast"]
        mfast_mean = statistics.mean(mfast)
        nfast_mean = statistics.mean(nfast)
        label = '`[0,1)`' if input_name == "random" else '`[-8,8)`'
        lines.append(
            f"| {label} | {mfast_mean:.3f} +/- "
            f"{statistics.stdev(mfast):.3f} | {nfast_mean:.3f} +/- "
            f"{statistics.stdev(nfast):.3f} | "
            f"{100.0 * (nfast_mean / mfast_mean - 1.0):+.3f}% |"
        )
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
