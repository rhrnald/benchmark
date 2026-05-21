#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "0.attention"
SRC = OUT_DIR / "main.cu"

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

SMOKE_COMMITS = [(6, 6), (8, 8)]
COARSE_COMMITS = [(6, 6), (6, 5), (6, 4), (7, 6), (8, 4), (8, 8), (5, 6), (4, 4)]
SMOKE_MASKS = [0x00F, 0x01F]
COARSE_MASKS = [
    0x00F,
    0x01F,
    0x02F,
    0x03F,
    0x04F,
    0x08F,
    0x0CF,
    0x10F,
    0x20F,
    0x30F,
    0x40F,
    0x80F,
    0xC0F,
    0x100F,
    0x200F,
    0x300F,
    0x400F,
    0x800F,
    0xC00F,
    0x3000F,
    0x3C00F,
]
SMOKE_REGS = [(120, 192)]
COARSE_REGS = [(120, 192), (128, 184), (120, 184)]

BASE_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_100a,code=sm_100a",
    "--expt-relaxed-constexpr",
    "-Xptxas=-v",
]

SUMMARY_FIELDS = [
    "tag",
    "qk_commit",
    "pv_commit",
    "dep_mask",
    "dep_bits",
    "producer_regs",
    "consumer_regs",
    "compile_rc",
    "run_status",
    "kernel_status",
    "cuda_error",
    "elapsed_ms_values",
    "median_elapsed_ms",
    "tflops_values",
    "median_tflops",
    "validation_status",
    "ptxas_regs",
    "spill_stores",
    "spill_loads",
    "seconds",
    "binary",
    "csvs",
    "build_log",
    "notes",
]


@dataclass(frozen=True)
class Candidate:
    qk_commit: int
    pv_commit: int
    dep_mask: int
    producer_regs: int
    consumer_regs: int

    @property
    def tag(self) -> str:
        return (
            f"qk{self.qk_commit}_pv{self.pv_commit}_"
            f"dep{self.dep_mask:05x}_p{self.producer_regs}_c{self.consumer_regs}"
        )


def rel_or_abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else OUT_DIR / p


def parse_pair_list(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.replace("x", ":").split(":", 1)
        pairs.append((int(left, 0), int(right, 0)))
    return pairs


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item, 0))
    return values


def dep_bits(mask: int) -> str:
    names = [name for bit, name in enumerate(BIT_NAMES) if mask & (1 << bit)]
    return "|".join(names) if names else "none"


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    if args.mode == "smoke":
        commits = SMOKE_COMMITS
        masks = SMOKE_MASKS
        regs = SMOKE_REGS
    elif args.mode == "coarse":
        commits = COARSE_COMMITS
        masks = COARSE_MASKS
        regs = COARSE_REGS
    else:
        commits = parse_pair_list(args.commit_pairs)
        masks = parse_int_list(args.masks)
        regs = parse_pair_list(args.reg_pairs)

    cases = [
        Candidate(qk, pv, mask, preg, creg)
        for qk, pv in commits
        for mask in masks
        for preg, creg in regs
    ]
    return cases[args.start_index :]


def read_done_tags(summary: Path) -> set[str]:
    if not summary.exists():
        return set()
    with summary.open(newline="") as f:
        return {row["tag"] for row in csv.DictReader(f) if row.get("tag")}


def append_row(summary: Path, row: dict) -> None:
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary.exists()
    with summary.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_benchmark_csv(path: Path) -> dict:
    if not path.exists():
        return {"kernel_status": "missing_csv"}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return {"kernel_status": "empty_csv"}
    parts = lines[-1].split(",")
    if len(parts) < 22:
        return {"kernel_status": "bad_csv"}
    return {
        "elapsed_ms": parts[-21],
        "total_tflops": parts[-4],
        "kernel_status": parts[-3],
        "cuda_error": parts[-2],
        "notes": parts[-1],
    }


def parse_validation_csv(path: Path) -> str:
    if not path.exists():
        return "missing_csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "empty_csv"
    return "ok" if all(row.get("status") == "ok" for row in rows) else "fail"


def parse_ptxas(log: str) -> tuple[str, str, str]:
    kernel_seen = False
    regs = spills_st = spills_ld = ""
    for line in log.splitlines():
        if "qk_tma_mma_ld_kernel" in line and "Function properties" in line:
            kernel_seen = True
            continue
        if kernel_seen and "bytes stack frame" in line:
            match = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if match:
                spills_st = match.group(2)
                spills_ld = match.group(3)
            continue
        if kernel_seen and "ptxas info" in line and "Used" in line and "registers" in line:
            match = re.search(r"Used (\d+) registers", line)
            if match:
                regs = match.group(1)
            break
    return regs, spills_st, spills_ld


def run_cmd(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def compile_candidate(args: argparse.Namespace, cand: Candidate, build_dir: Path) -> tuple[int, Path, Path, str, str, str]:
    case_dir = build_dir / cand.tag
    case_dir.mkdir(parents=True, exist_ok=True)
    binary = case_dir / "attention_custom_kernel"
    build_log = case_dir / "build.log"
    flags = [
        *BASE_FLAGS,
        f"-DATTENTION_QK_COMMIT_AFTER_ISSUES={cand.qk_commit}",
        f"-DATTENTION_PV_COMMIT_AFTER_ISSUES={cand.pv_commit}",
        "-DATTENTION_DEP_SWEEP=1",
        f"-DATTENTION_DEP_MASK=0x{cand.dep_mask:x}",
        f"-DATTENTION_SETMAXNREG_PRODUCER={cand.producer_regs}",
        f"-DATTENTION_SETMAXNREG_CONSUMER={cand.consumer_regs}",
    ]
    cmd = [args.nvcc, *flags, str(SRC.relative_to(ROOT)), "-o", str(binary), "-lcuda"]
    rc, log = run_cmd(cmd, ROOT, args.compile_timeout)
    build_log.write_text(log)
    regs, spills_st, spills_ld = parse_ptxas(log)
    return rc, binary, build_log, regs, spills_st, spills_ld


def benchmark_candidate(args: argparse.Namespace, cand: Candidate, binary: Path) -> tuple[str, list[str], list[str], str, str, list[Path], str]:
    elapsed: list[str] = []
    tflops: list[str] = []
    csvs: list[Path] = []
    run_status = "ok"
    kernel_status = ""
    cuda_error = ""
    notes = ""
    for rep in range(args.reps):
        csv_path = rel_or_abs(args.csv_template.format(tag=cand.tag, rep=rep))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        run_log = binary.parent / f"run_{rep}.log"
        cmd = [
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
        for attempt in range(args.busy_retries + 1):
            try:
                rc, log = run_cmd(cmd, ROOT, args.run_timeout)
            except subprocess.TimeoutExpired as exc:
                run_log.write_text(exc.stdout or "")
                return "timeout", elapsed, tflops, kernel_status, cuda_error, csvs, notes
            if "busy or unavailable" not in log:
                break
            time.sleep(2.0 + attempt)
        run_log.write_text(log)
        if rc != 0:
            run_status = f"failed:{rc}"
        parsed = parse_benchmark_csv(csv_path)
        csvs.append(csv_path)
        kernel_status = parsed.get("kernel_status", "")
        cuda_error = parsed.get("cuda_error", "")
        notes = parsed.get("notes", "")
        if parsed.get("elapsed_ms"):
            elapsed.append(parsed["elapsed_ms"])
        if parsed.get("total_tflops"):
            tflops.append(parsed["total_tflops"])
    return run_status, elapsed, tflops, kernel_status, cuda_error, csvs, notes


def validate_candidate(args: argparse.Namespace, cand: Candidate, binary: Path) -> str:
    if not args.validate_each:
        return ""
    csv_path = rel_or_abs(f"log/combo_sweep_validate_{cand.tag}.csv")
    cmd = [str(binary), "--validate-suite", "--csv", str(csv_path)]
    try:
        rc, log = run_cmd(cmd, ROOT, args.validate_timeout)
    except subprocess.TimeoutExpired as exc:
        (binary.parent / "validate.log").write_text(exc.stdout or "")
        return "timeout"
    (binary.parent / "validate.log").write_text(log)
    if rc != 0:
        return f"failed:{rc}"
    return parse_validation_csv(csv_path)


def median_text(values: list[str]) -> str:
    nums: list[float] = []
    for value in values:
        try:
            nums.append(float(value))
        except ValueError:
            pass
    return f"{statistics.median(nums):.3f}" if nums else ""


def run_case(args: argparse.Namespace, cand: Candidate, build_dir: Path) -> dict:
    start = time.time()
    row = {
        "tag": cand.tag,
        "qk_commit": cand.qk_commit,
        "pv_commit": cand.pv_commit,
        "dep_mask": f"0x{cand.dep_mask:05x}",
        "dep_bits": dep_bits(cand.dep_mask),
        "producer_regs": cand.producer_regs,
        "consumer_regs": cand.consumer_regs,
        "compile_rc": "",
        "run_status": "",
        "kernel_status": "",
        "cuda_error": "",
        "elapsed_ms_values": "",
        "median_elapsed_ms": "",
        "tflops_values": "",
        "median_tflops": "",
        "validation_status": "",
        "ptxas_regs": "",
        "spill_stores": "",
        "spill_loads": "",
        "seconds": "",
        "binary": "",
        "csvs": "",
        "build_log": "",
        "notes": "",
    }
    rc, binary, build_log, regs, spills_st, spills_ld = compile_candidate(args, cand, build_dir)
    row.update(
        {
            "compile_rc": rc,
            "binary": str(binary),
            "build_log": str(build_log),
            "ptxas_regs": regs,
            "spill_stores": spills_st,
            "spill_loads": spills_ld,
        }
    )
    if rc != 0:
        row["run_status"] = "compile_failed"
        row["seconds"] = f"{time.time() - start:.3f}"
        return row

    run_status, elapsed, tflops, kernel_status, cuda_error, csvs, notes = benchmark_candidate(args, cand, binary)
    row.update(
        {
            "run_status": run_status,
            "kernel_status": kernel_status,
            "cuda_error": cuda_error,
            "elapsed_ms_values": ";".join(elapsed),
            "median_elapsed_ms": median_text(elapsed),
            "tflops_values": ";".join(tflops),
            "median_tflops": median_text(tflops),
            "csvs": ";".join(str(p) for p in csvs),
            "notes": notes,
        }
    )
    row["validation_status"] = validate_candidate(args, cand, binary)
    row["seconds"] = f"{time.time() - start:.3f}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep early commit, dependency mask, and setmaxnreg knobs.")
    parser.add_argument("--mode", choices=["smoke", "coarse", "custom"], default="coarse")
    parser.add_argument("--commit-pairs", default="6:6,6:5,6:4,7:6,8:8")
    parser.add_argument("--masks", default="0x00f,0x00d,0x003,0x00c,0x000")
    parser.add_argument("--reg-pairs", default="120:192,128:184,120:184")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", default="log/combo_sweep_summary.csv")
    parser.add_argument("--csv-template", default="log/combo_sweep_{tag}_rep{rep}.csv")
    parser.add_argument("--build-dir", default="build/0.attention_combo_sweep")
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--k-tiles", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--run-timeout", type=int, default=30)
    parser.add_argument("--validate-timeout", type=int, default=180)
    parser.add_argument("--busy-retries", type=int, default=2)
    parser.add_argument("--validate-each", action="store_true")
    args = parser.parse_args()

    summary = rel_or_abs(args.summary)
    build_dir = ROOT / args.build_dir
    candidates = build_candidates(args)
    if args.max_cases > 0:
        candidates = candidates[: args.max_cases]
    done = read_done_tags(summary) if args.resume and not args.force else set()

    for idx, cand in enumerate(candidates):
        if cand.tag in done:
            print(f"[skip] {idx} {cand.tag}", flush=True)
            continue
        macros = (
            f"QK={cand.qk_commit} PV={cand.pv_commit} "
            f"dep=0x{cand.dep_mask:05x} regs={cand.producer_regs}/{cand.consumer_regs}"
        )
        if args.dry_run:
            print(f"[dry] {idx} {cand.tag} {macros}", flush=True)
            continue
        print(f"[run] {idx} {cand.tag} {macros}", flush=True)
        row = run_case(args, cand, build_dir)
        append_row(summary, row)
        print(
            f"[done] {cand.tag} compile={row['compile_rc']} run={row['run_status']} "
            f"kernel={row['kernel_status']} ms={row['median_elapsed_ms']} "
            f"TF={row['median_tflops']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.TimeoutExpired as exc:
        print(f"timeout: {exc}", file=sys.stderr)
        sys.exit(124)
