#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


SIZES = (8192, 16384, 32768)
SHAPES = ("4x16", "8x16", "4x32", "8x18", "12x12", "16x16")
SCHEDULERS = ("dynamic", "static")


def parse_one(path: Path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    return float(rows[0]["event_TFLOPS"])


def main() -> None:
    out_dir = Path(sys.argv[1])
    values: dict[tuple[int, str, str], list[float]] = {}
    for size in SIZES:
        for scheduler in SCHEDULERS:
            for shape in SHAPES:
                samples = [
                    parse_one(
                        out_dir
                        / "csv"
                        / f"{scheduler}_{shape}_{size}_p{run}.csv"
                    )
                    for run in (1, 2, 3)
                ]
                values[size, scheduler, shape] = samples

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "size",
                "scheduler",
                "shape",
                "sample1",
                "sample2",
                "sample3",
                "mean",
                "sample_sd",
                "static_vs_dynamic_pct",
            ]
        )
        for size in SIZES:
            for scheduler in SCHEDULERS:
                for shape in SHAPES:
                    samples = values[size, scheduler, shape]
                    dynamic_mean = statistics.mean(values[size, "dynamic", shape])
                    delta = (
                        100.0 * (statistics.mean(samples) / dynamic_mean - 1.0)
                        if scheduler == "static"
                        else 0.0
                    )
                    writer.writerow(
                        [
                            size,
                            scheduler,
                            shape,
                            *[f"{x:.3f}" for x in samples],
                            f"{statistics.mean(samples):.3f}",
                            f"{statistics.stdev(samples):.3f}",
                            f"{delta:+.4f}",
                        ]
                    )

    lines = [
        "# Recent direct N-split scheduler sweep",
        "",
        "BF16 uniform `[-8,8)`, W1/I5, three process samples per cell.",
        "",
    ]
    for size in SIZES:
        lines.extend(
            [
                f"## {size // 1024}K",
                "",
                "| shape | tasks/macro | dynamic | static | static vs dynamic |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for shape in SHAPES:
            m, n = map(int, shape.split("x"))
            dynamic = values[size, "dynamic", shape]
            static = values[size, "static", shape]
            dynamic_mean = statistics.mean(dynamic)
            static_mean = statistics.mean(static)
            delta = 100.0 * (static_mean / dynamic_mean - 1.0)
            lines.append(
                f"| `{shape}` | {m * n} | "
                f"{dynamic_mean:.3f} +/- {statistics.stdev(dynamic):.3f} | "
                f"{static_mean:.3f} +/- {statistics.stdev(static):.3f} | "
                f"{delta:+.3f}% |"
            )
        candidates = [
            (statistics.mean(values[size, scheduler, shape]), scheduler, shape)
            for scheduler in SCHEDULERS
            for shape in SHAPES
        ]
        best_mean, best_scheduler, best_shape = max(candidates)
        lines.extend(
            [
                "",
                f"Best: **{best_mean:.3f} TFLOP/s**, "
                f"`{best_scheduler} {best_shape}`.",
                "",
            ]
        )
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
