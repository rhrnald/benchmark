#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


VARIANTS = ("8x16", "16x8")
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
        variant: [
            read_tflops(out_dir / "csv" / f"static_{variant}_p{run}.csv")
            for run in PASSES
        ]
        for variant in VARIANTS
    }
    baseline = statistics.mean(values["8x16"])

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["variant", "sample1", "sample2", "sample3", "sample4",
             "mean", "sample_sd", "vs_8x16_pct"]
        )
        for variant in VARIANTS:
            samples = values[variant]
            mean = statistics.mean(samples)
            writer.writerow(
                [
                    variant,
                    *[f"{value:.3f}" for value in samples],
                    f"{mean:.3f}",
                    f"{statistics.stdev(samples):.3f}",
                    f"{100.0 * (mean / baseline - 1.0):+.4f}",
                ]
            )

    lines = [
        "# Static macro orientation comparison",
        "",
        "16K square BF16 GEMM, uniform `[-8,8)`, FP32 accumulation/output,",
        "148 persistent CTAs, static grid-stride ownership, W1/I5, four",
        "ABBA-position-balanced independent process samples per variant.",
        "",
        "| static macro | samples (TFLOP/s) | mean +/- sample SD | vs `8x16` |",
        "|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        samples = values[variant]
        mean = statistics.mean(samples)
        sample_text = " / ".join(f"{value:.3f}" for value in samples)
        lines.append(
            f"| `{variant}` | {sample_text} | "
            f"**{mean:.3f} +/- {statistics.stdev(samples):.3f}** | "
            f"{100.0 * (mean / baseline - 1.0):+.3f}% |"
        )
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
