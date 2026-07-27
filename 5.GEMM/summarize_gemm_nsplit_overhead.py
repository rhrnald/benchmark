#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


VARIANTS = (
    "baseline",
    "no_memset",
    "sink_keep",
    "sink_trim",
    "fixed",
    "suspend",
    "fixed_sink",
    "all",
)
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
        (input_name, variant): [
            read_tflops(out_dir / "csv" / f"{input_name}_{variant}_p{run}.csv")
            for run in PASSES
        ]
        for input_name in INPUTS
        for variant in VARIANTS
    }

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input",
                "variant",
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "mean",
                "sample_sd",
                "paired_delta_mean_pct",
                "paired_delta_sd_pct",
            ]
        )
        for input_name in INPUTS:
            baseline = values[input_name, "baseline"]
            for variant in VARIANTS:
                samples = values[input_name, variant]
                deltas = [
                    100.0 * (value / base - 1.0)
                    for value, base in zip(samples, baseline)
                ]
                writer.writerow(
                    [
                        input_name,
                        variant,
                        *[f"{value:.3f}" for value in samples],
                        f"{statistics.mean(samples):.3f}",
                        f"{statistics.stdev(samples):.3f}",
                        f"{statistics.mean(deltas):+.4f}",
                        f"{statistics.stdev(deltas):.4f}",
                    ]
                )

    lines = [
        "# Static 8x16 overhead ablation",
        "",
        "16K square BF16 GEMM, FP32 accumulation/output, 148 persistent CTAs,",
        "static 8x16 M-fast scheduling, W1/I5, four position-balanced process",
        "samples per cell.",
        "",
    ]
    for input_name in INPUTS:
        label = '`[0,1)`' if input_name == "random" else '`[-8,8)`'
        baseline = values[input_name, "baseline"]
        lines.extend(
            [
                f"## {label}",
                "",
                "| variant | TFLOP/s mean +/- SD | paired delta vs baseline |",
                "|---|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            samples = values[input_name, variant]
            deltas = [
                100.0 * (value / base - 1.0)
                for value, base in zip(samples, baseline)
            ]
            lines.append(
                f"| `{variant}` | {statistics.mean(samples):.3f} +/- "
                f"{statistics.stdev(samples):.3f} | "
                f"{statistics.mean(deltas):+.3f}% +/- "
                f"{statistics.stdev(deltas):.3f}%p |"
            )
        lines.append("")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
