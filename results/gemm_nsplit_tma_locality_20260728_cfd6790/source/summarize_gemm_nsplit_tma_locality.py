#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


INPUTS = ("random", "random-signed8")
MODES = ("dense", "same_a", "same_b", "same")
PASSES = range(1, 5)
SIZE = 16384
LOGICAL_BYTES = (SIZE // 256) ** 2 * (SIZE // 64) * 65536


def read_ms(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    return float(rows[0]["event_ms"])


def main() -> None:
    out_dir = Path(sys.argv[1])
    values: dict[tuple[str, str], list[float]] = {}
    for input_name in INPUTS:
        for mode in MODES:
            values[input_name, mode] = [
                read_ms(out_dir / "csv" / f"{input_name}_{mode}_p{run}.csv")
                for run in PASSES
            ]

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input",
                "address_mode",
                "sample1_ms",
                "sample2_ms",
                "sample3_ms",
                "sample4_ms",
                "mean_ms",
                "sample_sd_ms",
                "logical_TBps",
                "speedup_vs_dense",
            ]
        )
        for input_name in INPUTS:
            dense_mean = statistics.mean(values[input_name, "dense"])
            for mode in MODES:
                samples = values[input_name, mode]
                mean_ms = statistics.mean(samples)
                writer.writerow(
                    [
                        input_name,
                        mode,
                        *[f"{x:.6f}" for x in samples],
                        f"{mean_ms:.6f}",
                        f"{statistics.stdev(samples):.6f}",
                        f"{LOGICAL_BYTES / (mean_ms * 1e-3) / 1e12:.3f}",
                        f"{dense_mean / mean_ms:.4f}",
                    ]
                )

    labels = {"random": "`[0,1)`", "random-signed8": "`[-8,8)`"}
    lines = [
        "# 16K TMA-only address-locality ablation",
        "",
        "All modes issue the same 64 GiB logical A/B payload. Mean +/- sample",
        "SD across four independent W1/I5 processes; method order is fully",
        "position-balanced.",
        "",
        "| input | address mode | ms | logical TB/s | speedup vs dense |",
        "|---|---|---:|---:|---:|",
    ]
    for input_name in INPUTS:
        dense_mean = statistics.mean(values[input_name, "dense"])
        for mode in MODES:
            samples = values[input_name, mode]
            mean_ms = statistics.mean(samples)
            lines.append(
                f"| {labels[input_name]} | `{mode}` | "
                f"{mean_ms:.6f} +/- {statistics.stdev(samples):.6f} | "
                f"{LOGICAL_BYTES / (mean_ms * 1e-3) / 1e12:.3f} | "
                f"{dense_mean / mean_ms:.4f}x |"
            )
    lines.append("")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
