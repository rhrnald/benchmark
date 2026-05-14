#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


BIT_NAMES = [
    "qk_p1_wait_p0",
    "qk_p0_wait_p1",
    "pv_p1_wait_p0",
    "pv_p0_wait_p1",
    "vtma_wait_qk",
    "vtma_wait_k",
    "sum_h0_wait_v",
    "sum_h1_wait_v",
    "next_ktma_wait_s1",
    "next_ktma_wait_pv",
    "ktma_p1_wait_p0",
    "ktma_p0_wait_p1",
    "vtma_p1_wait_p0",
    "vtma_p0_wait_p1",
    "sum_h0_p1_wait_p0",
    "sum_h0_p0_wait_p1",
    "sum_h1_p1_wait_p0",
    "sum_h1_p0_wait_p1",
]

BASELINE_MASK = 0x00D
CSV_FIELDS = [
    "round",
    "mask",
    "mask_hex",
    "enabled_bits",
    "elapsed_ms",
    "total_TFLOP_per_s",
    "kernel_status",
    "cuda_error",
    "build_status",
    "run_status",
    "seconds",
    "csv",
    "binary",
    "notes",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def enabled_bits(mask: int) -> str:
    return "|".join(name for bit, name in enumerate(BIT_NAMES) if mask & (1 << bit))


def benchmark_csv_path(root: Path, label: str) -> Path:
    return root / "0.attention" / f"attention_dep_sweep_{label}.csv"


def read_benchmark_csv(path: Path) -> dict:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return {}
    # The benchmark's shape fields contain unquoted commas, so parse the stable
    # tail fields from the right instead of using csv.DictReader.
    parts = lines[1].split(",")
    if len(parts) < 21:
        return {}
    return {
        "elapsed_ms": parts[-21],
        "qk_TFLOP_per_s": parts[-6],
        "pv_TFLOP_per_s": parts[-5],
        "total_TFLOP_per_s": parts[-4],
        "status": parts[-3],
        "cuda_error": parts[-2],
        "notes": parts[-1],
    }


def build_cmd(args: argparse.Namespace, root: Path, binary: Path, mask: int | None) -> list[str]:
    cmd = [
        args.nvcc,
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--expt-relaxed-constexpr",
        "-Xptxas=-v",
        f"-DATTENTION_PRODUCER_REGS={args.producer_regs}",
        f"-DATTENTION_CONSUMER_REGS={args.consumer_regs}",
        "-DATTENTION_STORE_OUTPUT=1",
    ]
    if mask is not None:
        cmd += [
            "-DATTENTION_DEP_SWEEP=1",
            f"-DATTENTION_DEP_MASK={mask}",
        ]
    cmd += [
        "0.attention/attention_fused_real_attention.cu",
        "-o",
        str(binary),
        "-lcuda",
    ]
    return cmd


def run_cmd(binary: Path, csv_path: Path, args: argparse.Namespace) -> list[str]:
    return [
        str(binary),
        "--blocks",
        str(args.blocks),
        "--k-tiles",
        str(args.k_tiles),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--csv",
        str(csv_path),
    ]


def run_case(root: Path, args: argparse.Namespace, round_name: str, mask: int | None) -> dict:
    label = "default" if mask is None else f"mask_0x{mask:03x}"
    build_dir = root / args.build_dir / label
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "attention_dep_sweep"
    csv_path = benchmark_csv_path(root, label)
    build_log = build_dir / "build.log"
    run_log = build_dir / "run.log"

    row = {
        "round": round_name,
        "mask": "-1" if mask is None else str(mask),
        "mask_hex": "default" if mask is None else f"0x{mask:03x}",
        "enabled_bits": "default" if mask is None else enabled_bits(mask),
        "elapsed_ms": "",
        "total_TFLOP_per_s": "",
        "kernel_status": "",
        "cuda_error": "",
        "build_status": "ok",
        "run_status": "not_run",
        "seconds": "",
        "csv": str(csv_path),
        "binary": str(binary),
        "notes": "",
    }

    start = time.time()
    try:
        build = subprocess.run(
            build_cmd(args, root, binary, mask),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.build_timeout,
        )
        build_log.write_text(build.stdout)
    except subprocess.TimeoutExpired as exc:
        build_log.write_text(exc.stdout or "")
        row["build_status"] = "timeout"
        row["seconds"] = f"{time.time() - start:.3f}"
        return row

    if build.returncode != 0:
        row["build_status"] = f"failed:{build.returncode}"
        row["seconds"] = f"{time.time() - start:.3f}"
        return row

    try:
        run = subprocess.run(
            run_cmd(binary, csv_path, args),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.run_timeout,
        )
        run_log.write_text(run.stdout)
    except subprocess.TimeoutExpired as exc:
        run_log.write_text(exc.stdout or "")
        row["run_status"] = "timeout"
        row["seconds"] = f"{time.time() - start:.3f}"
        return row

    row["run_status"] = "ok" if run.returncode == 0 else f"failed:{run.returncode}"
    if csv_path.exists():
        bench = read_benchmark_csv(csv_path)
        row["elapsed_ms"] = bench.get("elapsed_ms", "")
        row["total_TFLOP_per_s"] = bench.get("total_TFLOP_per_s", "")
        row["kernel_status"] = bench.get("status", "")
        row["cuda_error"] = bench.get("cuda_error", "")
        row["notes"] = bench.get("notes", "")
    row["seconds"] = f"{time.time() - start:.3f}"
    return row


def round_masks(round_name: str) -> list[int | None]:
    if round_name == "round0":
        return [None, BASELINE_MASK]
    if round_name == "round1":
        masks = [BASELINE_MASK]
        masks += [BASELINE_MASK & ~(1 << bit) for bit in range(4)]
        masks += [BASELINE_MASK | (1 << bit) for bit in range(4, 10)]
        return list(dict.fromkeys(masks))
    if round_name == "sameunit":
        masks = [BASELINE_MASK]
        masks += [BASELINE_MASK | (1 << bit) for bit in range(10, len(BIT_NAMES))]
        return list(dict.fromkeys(masks))
    if round_name == "full":
        return list(range(1024))
    raise ValueError(f"unsupported round without dynamic runner: {round_name}")


def parse_mask_list(mask_list: str) -> list[int | None]:
    masks: list[int | None] = []
    for item in mask_list.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "default":
            masks.append(None)
        else:
            masks.append(int(item, 0))
    return masks


def append_rows(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            f.flush()


def parse_score(row: dict) -> float:
    if row.get("run_status") != "ok" or row.get("kernel_status") != "ok":
        return float("-inf")
    try:
        return float(row.get("total_TFLOP_per_s") or "-inf")
    except ValueError:
        return float("-inf")


def run_static_round(root: Path, args: argparse.Namespace) -> None:
    masks = parse_mask_list(args.masks) if args.masks else round_masks(args.round)
    for mask in masks:
        row = run_case(root, args, args.round, mask)
        append_rows(root / args.output, [row])
        print(
            f"{row['round']} {row['mask_hex']} build={row['build_status']} "
            f"run={row['run_status']} tflops={row['total_TFLOP_per_s']}",
            flush=True,
        )


def run_beam(root: Path, args: argparse.Namespace) -> None:
    seen: set[int] = set()
    frontier = [BASELINE_MASK]
    for depth in range(args.beam_depth):
        candidates: list[int] = []
        for mask in frontier:
            candidates.append(mask)
            for bit in range(len(BIT_NAMES)):
                candidates.append(mask ^ (1 << bit))
        candidates = [m for m in dict.fromkeys(candidates) if m not in seen]
        rows = []
        for mask in candidates:
            seen.add(mask)
            row = run_case(root, args, f"beam{depth}", mask)
            rows.append(row)
            append_rows(root / args.output, [row])
            print(
                f"beam{depth} 0x{mask:03x} build={row['build_status']} "
                f"run={row['run_status']} tflops={row['total_TFLOP_per_s']}",
                flush=True,
            )
        scored = sorted(rows, key=parse_score, reverse=True)
        frontier = [int(row["mask"]) for row in scored[: args.beam_width] if parse_score(row) > float("-inf")]
        if not frontier:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile-time attention dependency sweep")
    parser.add_argument("--round", choices=["round0", "round1", "sameunit", "beam", "full"], default="round1")
    parser.add_argument("--masks", default="", help="Comma-separated mask list, e.g. default,0x00f,0x00d")
    parser.add_argument("--output", default="0.attention/attention_dep_sweep_results.csv")
    parser.add_argument("--build-dir", default="build_attention_dep_sweep")
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--producer-regs", type=int, default=128)
    parser.add_argument("--consumer-regs", type=int, default=184)
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--k-tiles", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--build-timeout", type=int, default=180)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-depth", type=int, default=4)
    args = parser.parse_args()

    root = repo_root()
    if args.round == "beam":
        run_beam(root, args)
    else:
        run_static_round(root, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
