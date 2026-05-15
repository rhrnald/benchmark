#!/usr/bin/env python3
import argparse
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
