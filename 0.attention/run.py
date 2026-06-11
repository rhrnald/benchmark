#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLOTTER = HERE / "plot_attention_trace.py"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def run_checked(cmd: list[str], cwd: Path = HERE) -> None:
    print(" ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_tail_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"status": "missing_csv"}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return {"status": "empty_csv"}
    parts = lines[-1].split(",")
    if len(parts) < 4:
        return {"status": "bad_csv"}
    return {
        "elapsed_ms": parts[-21] if len(parts) >= 21 else "",
        "total_tflops": parts[-4],
        "status": parts[-3],
        "cuda_error": parts[-2],
    }


def benchmark(args: argparse.Namespace) -> int:
    csv_path = resolve(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked([
        resolve(args.binary),
        "--blocks",
        args.blocks,
        "--k-tiles",
        args.k_tiles,
        "--warmup",
        args.warmup,
        "--iters",
        args.iters,
        "--csv",
        csv_path,
    ])
    parsed = parse_tail_csv(csv_path)
    print(
        "total_TFLOP_per_s={total_tflops} elapsed_ms={elapsed_ms} "
        "status={status} cuda_error={cuda_error} csv={csv}".format(
            csv=csv_path, **parsed
        )
    )
    return 0 if parsed.get("status") == "ok" else 1


def plot(args: argparse.Namespace) -> int:
    trace_csv = resolve(args.trace_csv)
    summary_csv = resolve(args.summary_csv)
    svg = resolve(args.svg)
    for path in (trace_csv, summary_csv, svg):
        path.parent.mkdir(parents=True, exist_ok=True)

    run_checked([
        resolve(args.binary),
        "--blocks",
        args.blocks,
        "--k-tiles",
        args.k_tiles,
        "--warmup",
        args.warmup,
        "--iters",
        args.iters,
        "--clock-trace",
        "--clock-trace-start",
        args.clock_trace_start,
        "--clock-trace-iters",
        args.clock_trace_iters,
        "--csv",
        trace_csv,
    ])
    parsed = parse_tail_csv(trace_csv)
    if parsed.get("status") != "ok":
        print(f"trace_status={parsed.get('status')} csv={trace_csv}", file=sys.stderr)
        return 1

    title = (
        "Attention best trace, "
        f"iter{args.clock_trace_start}-"
        f"{int(args.clock_trace_start) + int(args.clock_trace_iters) - 1}"
    )
    run_checked([
        sys.executable,
        PLOTTER,
        "--trace",
        trace_csv,
        "--iter-start",
        args.clock_trace_start,
        "--num-iters",
        args.clock_trace_iters,
        "--summary-csv",
        summary_csv,
        "--svg",
        svg,
        "--title",
        title,
    ])
    print(f"svg={svg}")
    print(f"summary_csv={summary_csv}")
    print(f"trace_csv={trace_csv}")
    return 0


def _int_field(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return 0
    return int(value)


def _normal_lt_zero(mean: float, stddev: float) -> float:
    if stddev <= 0.0:
        return 1.0 if mean < 0.0 else 0.0
    return 0.5 * math.erfc(mean / (stddev * math.sqrt(2.0)))


def _mean_std(values: list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _parse_starts(starts: str) -> list[int]:
    parsed = [int(part.strip()) for part in starts.split(",") if part.strip()]
    if not parsed:
        raise SystemExit("--starts must contain at least one iteration")
    return parsed


def _parse_trace_sample(trace_csv: Path, trace_start: int, sample: int) -> dict[str, object]:
    with trace_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one trace row in {trace_csv}, got {len(rows)}")
    row = rows[0]
    pack_starts = [_int_field(row, f"pack_warp{i}_start") for i in range(8)]
    nonzero_pack_starts = [
        (idx, value) for idx, value in enumerate(pack_starts) if value > 0
    ]
    earliest_pack_lane, earliest_pack_start = (
        min(nonzero_pack_starts, key=lambda item: item[1])
        if nonzero_pack_starts
        else (-1, _int_field(row, "pack_start"))
    )
    early_wait_warp = earliest_pack_lane // 2 if earliest_pack_lane >= 0 else -1
    early_wait_start = _int_field(row, "early_wait_start")
    early_wait_end = _int_field(row, "early_wait_end")
    if early_wait_warp >= 0:
        early_wait_start = early_wait_start or _int_field(
            row, f"qk_wait_warp{early_wait_warp}_start"
        )
        early_wait_end = early_wait_end or _int_field(
            row, f"qk_wait_warp{early_wait_warp}_end"
        )
    full_issue_end = _int_field(row, "full_issue_end")
    full_done_end = _int_field(row, "full_done_end")
    softmax_start_after_early_wait = (
        earliest_pack_start - early_wait_end if earliest_pack_start and early_wait_end else 0
    )
    mma_done_after_early_wait = (
        full_done_end - early_wait_end if full_done_end and early_wait_end else 0
    )
    hazard_margin_after_early_wait = (
        earliest_pack_start - full_done_end if earliest_pack_start and full_done_end else 0
    )
    softmax_start_after_full_issue = (
        earliest_pack_start - full_issue_end if earliest_pack_start and full_issue_end else 0
    )
    mma_done_after_full_issue = (
        full_done_end - full_issue_end if full_done_end and full_issue_end else 0
    )
    hazard_margin = earliest_pack_start - full_done_end if earliest_pack_start and full_done_end else 0

    parsed: dict[str, object] = {
        "trace_start": trace_start,
        "sample": sample,
        "status": row.get("status", ""),
        "cuda_error": row.get("cuda_error", ""),
        "early_commit_end": _int_field(row, "early_commit_end"),
        "full_issue_end": full_issue_end,
        "full_done_end": full_done_end,
        "full_done_wait_cycles": _int_field(row, "full_done_wait_cycles"),
        "early_wait_start": early_wait_start,
        "early_wait_end": early_wait_end,
        "early_wait_cycles": _int_field(row, "early_wait_cycles"),
        "earliest_pack_start": earliest_pack_start,
        "earliest_pack_lane": earliest_pack_lane,
        "softmax_start_after_early_wait": softmax_start_after_early_wait,
        "mma_done_after_early_wait": mma_done_after_early_wait,
        "hazard_margin_after_early_wait": hazard_margin_after_early_wait,
        "softmax_start_after_full_issue": softmax_start_after_full_issue,
        "mma_done_after_full_issue": mma_done_after_full_issue,
        "hazard_margin": hazard_margin,
        "trace_csv": str(trace_csv),
    }
    for i, value in enumerate(pack_starts):
        parsed[f"pack_warp{i}_start"] = value
    for i in range(4):
        parsed[f"qk_wait_warp{i}_start"] = _int_field(row, f"qk_wait_warp{i}_start")
        parsed[f"qk_wait_warp{i}_end"] = _int_field(row, f"qk_wait_warp{i}_end")
    return parsed


def _write_early_commit_summary(out_dir: Path, samples: list[dict[str, object]]) -> None:
    sample_fields = [
        "trace_start",
        "sample",
        "status",
        "cuda_error",
        "early_commit_end",
        "full_issue_end",
        "full_done_end",
        "full_done_wait_cycles",
        "early_wait_start",
        "early_wait_end",
        "early_wait_cycles",
        "earliest_pack_start",
        "earliest_pack_lane",
        "softmax_start_after_early_wait",
        "mma_done_after_early_wait",
        "hazard_margin_after_early_wait",
        "softmax_start_after_full_issue",
        "mma_done_after_full_issue",
        "hazard_margin",
    ] + [f"pack_warp{i}_start" for i in range(8)]
    sample_fields += [
        field
        for i in range(4)
        for field in (f"qk_wait_warp{i}_start", f"qk_wait_warp{i}_end")
    ]
    sample_fields += ["trace_csv"]

    samples_csv = out_dir / "early_commit_samples.csv"
    with samples_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(samples)

    summary_fields = [
        "trace_start",
        "n",
        "valid_n",
        "collision_count",
        "empirical_collision_rate",
        "x_mean_mma_done_after_early_wait",
        "x_stddev_early_wait",
        "y_mean_softmax_start_after_early_wait",
        "y_stddev_early_wait",
        "hazard_margin_after_early_wait_mean",
        "hazard_margin_after_early_wait_stddev",
        "x_mean_mma_done_after_full_issue",
        "x_stddev_full_issue",
        "y_mean_softmax_start_after_full_issue",
        "y_stddev_full_issue",
        "independent_normal_collision_prob",
        "paired_margin_normal_collision_prob",
    ]
    grouped = sorted({int(row["trace_start"]) for row in samples})
    summary_rows: list[dict[str, object]] = []
    for trace_start in grouped:
        rows = [row for row in samples if int(row["trace_start"]) == trace_start]
        valid = [
            row for row in rows
            if row["status"] == "ok"
            and row["cuda_error"] == "no error"
            and int(row["early_wait_end"]) > 0
            and int(row["full_done_end"]) > 0
            and int(row["earliest_pack_start"]) > 0
        ]
        x_values = [int(row["mma_done_after_early_wait"]) for row in valid]
        y_values = [int(row["softmax_start_after_early_wait"]) for row in valid]
        margins = [int(row["hazard_margin_after_early_wait"]) for row in valid]
        x_mean, x_std = _mean_std(x_values)
        y_mean, y_std = _mean_std(y_values)
        margin_mean, margin_std = _mean_std(margins)
        x_full_values = [int(row["mma_done_after_full_issue"]) for row in valid]
        y_full_values = [int(row["softmax_start_after_full_issue"]) for row in valid]
        x_full_mean, x_full_std = _mean_std(x_full_values)
        y_full_mean, y_full_std = _mean_std(y_full_values)
        independent_std = math.sqrt(x_std * x_std + y_std * y_std)
        independent_p = _normal_lt_zero(y_mean - x_mean, independent_std)
        paired_p = _normal_lt_zero(margin_mean, margin_std)
        collision_count = sum(1 for margin in margins if margin < 0)
        summary_rows.append({
            "trace_start": trace_start,
            "n": len(rows),
            "valid_n": len(valid),
            "collision_count": collision_count,
            "empirical_collision_rate": collision_count / len(valid) if valid else 0.0,
            "x_mean_mma_done_after_early_wait": x_mean,
            "x_stddev_early_wait": x_std,
            "y_mean_softmax_start_after_early_wait": y_mean,
            "y_stddev_early_wait": y_std,
            "hazard_margin_after_early_wait_mean": margin_mean,
            "hazard_margin_after_early_wait_stddev": margin_std,
            "x_mean_mma_done_after_full_issue": x_full_mean,
            "x_stddev_full_issue": x_full_std,
            "y_mean_softmax_start_after_full_issue": y_full_mean,
            "y_stddev_full_issue": y_full_std,
            "independent_normal_collision_prob": independent_p,
            "paired_margin_normal_collision_prob": paired_p,
        })

    summary_csv = out_dir / "early_commit_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = out_dir / "early_commit_summary.md"
    with summary_md.open("w") as f:
        f.write("# Early Commit Safety Measurement\n\n")
        f.write("Primary softmax/store start metric: earliest nonzero `pack_warp*_start` per traced iteration.\n\n")
        f.write("Primary X/Y anchor: consumer `qk_done` early-commit wait end for the same warp as the earliest pack lane.\n\n")
        f.write("| trace_start | valid/n | empirical collision | independent normal | paired margin normal | X mean | Y mean | margin mean |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            f.write(
                "| {trace_start} | {valid_n}/{n} | {empirical_collision_rate:.6g} | "
                "{independent_normal_collision_prob:.6g} | {paired_margin_normal_collision_prob:.6g} | "
                "{x_mean_mma_done_after_early_wait:.3f} | {y_mean_softmax_start_after_early_wait:.3f} | "
                "{hazard_margin_after_early_wait_mean:.3f} |\n".format(**row)
            )


def early_commit_measure(args: argparse.Namespace) -> int:
    starts = _parse_starts(args.starts)
    out_dir = resolve(args.out_dir)
    trace_root = out_dir / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, object]] = []
    for trace_start in starts:
        start_dir = trace_root / f"start_{trace_start}"
        start_dir.mkdir(parents=True, exist_ok=True)
        for sample in range(1, int(args.samples) + 1):
            trace_csv = start_dir / f"sample_{sample:04d}.csv"
            run_checked([
                resolve(args.binary),
                "--blocks",
                args.blocks,
                "--k-tiles",
                args.k_tiles,
                "--warmup",
                args.warmup,
                "--iters",
                "1",
                "--clock-trace",
                "--clock-trace-start",
                str(trace_start),
                "--clock-trace-iters",
                "1",
                "--csv",
                trace_csv,
            ])
            samples.append(_parse_trace_sample(trace_csv, trace_start, sample))
            latest = samples[-1]
            print(
                "trace_start={trace_start} sample={sample} "
                "mma_done_after_early_wait={mma_done_after_early_wait} "
                "softmax_start_after_early_wait={softmax_start_after_early_wait} "
                "hazard_margin={hazard_margin_after_early_wait} "
                "status={status} cuda_error={cuda_error}".format(**latest),
                flush=True,
            )

    _write_early_commit_summary(out_dir, samples)
    print(f"samples_csv={out_dir / 'early_commit_samples.csv'}")
    print(f"summary_csv={out_dir / 'early_commit_summary.csv'}")
    print(f"summary_md={out_dir / 'early_commit_summary.md'}")
    return 0


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", required=True)
    parser.add_argument("--blocks", default="4096")
    parser.add_argument("--k-tiles", default="256")
    parser.add_argument("--warmup", default="3")
    parser.add_argument("--iters", default="10")


def main() -> int:
    parser = argparse.ArgumentParser(description="Attention benchmark runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bench = sub.add_parser("benchmark", help="run the fastest benchmark path")
    add_common_run_args(bench)
    bench.add_argument("--csv", default="log/best.csv")
    bench.set_defaults(func=benchmark)

    plot_cmd = sub.add_parser("plot", help="run a clock trace and draw the cycle SVG")
    add_common_run_args(plot_cmd)
    plot_cmd.add_argument("--clock-trace-start", default="56")
    plot_cmd.add_argument("--clock-trace-iters", default="8")
    plot_cmd.add_argument("--trace-csv", default="log/best_trace.csv")
    plot_cmd.add_argument("--summary-csv", default="log/best_trace_summary.csv")
    plot_cmd.add_argument("--svg", default="log/best.svg")
    plot_cmd.set_defaults(func=plot)

    early = sub.add_parser("early-commit-measure", help="sample early-commit hazard timing")
    add_common_run_args(early)
    early.add_argument("--starts", default="30,40,50,60")
    early.add_argument("--samples", type=int, default=100)
    early.add_argument("--out-dir", default="log/early_commit_measure")
    early.set_defaults(func=early_commit_measure)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
