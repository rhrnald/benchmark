#!/usr/bin/env python3
import csv
import re
import statistics
import sys
from pathlib import Path


SIZES = (8192, 16384, 32768)
OURS_BY_SIZE = {
    8192: ("ours_16x16", "ours_12x12", "ours_8x18"),
    16384: ("ours_16x16",),
    32768: ("ours_16x16", "ours_12x12", "ours_8x18"),
}


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
    values: dict[tuple[int, str], list[float]] = {}

    for size in SIZES:
        for method in (*OURS_BY_SIZE[size], "cublas", "cutlass"):
            samples = []
            for run in (1, 2, 3):
                if method.startswith("ours_"):
                    samples.append(
                        parse_ours(out_dir / "csv" / f"{method}_{size}_p{run}.csv")
                    )
                else:
                    samples.append(
                        parse_library(
                            out_dir / "logs" / f"{method}_{size}_p{run}.log",
                            method,
                        )
                    )
            values[size, method] = samples

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["size", "method", "sample1", "sample2", "sample3", "mean", "sample_sd"]
        )
        for size in SIZES:
            for method in (*OURS_BY_SIZE[size], "cublas", "cutlass"):
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

    winners = {
        size: max(
            OURS_BY_SIZE[size],
            key=lambda method: statistics.mean(values[size, method]),
        )
        for size in SIZES
    }

    lines = [
        "# Recent direct N-split BF16 uniform `[-8,8)` comparison",
        "",
        "Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.",
        "Each cell is mean +/- sample SD across three independent processes;",
        "each process used one warmup and five timed launches.",
        "",
        "## Ours scheduler selection",
        "",
        "| size | macro | TFLOP/s |",
        "|---:|---:|---:|",
    ]
    for size in SIZES:
        for method in OURS_BY_SIZE[size]:
            samples = values[size, method]
            winner = " **selected**" if method == winners[size] else ""
            lines.append(
                f"| {size // 1024}K | `{method.removeprefix('ours_')}` | "
                f"{statistics.mean(samples):.3f} +/- "
                f"{statistics.stdev(samples):.3f}{winner} |"
            )

    lines.extend(
        [
            "",
            "## Selected comparison",
            "",
            "| size | CUTLASS | cuBLAS | ours (recent N-split best) | ours vs cuBLAS |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for size in SIZES:
        winner = winners[size]
        means = {
            method: statistics.mean(values[size, method])
            for method in (winner, "cublas", "cutlass")
        }
        sds = {
            method: statistics.stdev(values[size, method])
            for method in (winner, "cublas", "cutlass")
        }
        delta = 100.0 * (means[winner] / means["cublas"] - 1.0)
        lines.append(
            f"| {size // 1024}K | "
            f"{means['cutlass']:.3f} +/- {sds['cutlass']:.3f} | "
            f"{means['cublas']:.3f} +/- {sds['cublas']:.3f} | "
            f"{means[winner]:.3f} +/- {sds[winner]:.3f} "
            f"(`{winner.removeprefix('ours_')}`) | {delta:+.3f}% |"
        )
    lines.append("")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
