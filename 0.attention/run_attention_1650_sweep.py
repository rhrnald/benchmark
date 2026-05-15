#!/usr/bin/env python3
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "0.attention" / "attention_fused_real_attention.cu"
OUT_DIR = ROOT / "0.attention"
BIN_REL = Path("0.attention") / "attention_custom_kernel"

BASE_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_100a,code=sm_100a",
    "--expt-relaxed-constexpr",
    "-Xptxas=-v",
    "-DATTENTION_STORE_OUTPUT=1",
]

PERF_ARGS = [
    "--blocks",
    "4096",
    "--k-tiles",
    "256",
    "--warmup",
    "3",
    "--iters",
    "10",
]


@dataclass(frozen=True)
class Candidate:
    tag: str
    macros: tuple[tuple[str, str], ...]


CANDIDATES = [
    Candidate("baseline_default", ()),
    Candidate("dep_00d_p120_c184", (("ATTENTION_DEP_SWEEP", "1"),
                                     ("ATTENTION_DEP_MASK", "0x00d"),
                                     ("ATTENTION_PRODUCER_REGS", "120"),
                                     ("ATTENTION_CONSUMER_REGS", "184"))),
    Candidate("dep_00f_p120_c184", (("ATTENTION_DEP_SWEEP", "1"),
                                     ("ATTENTION_DEP_MASK", "0x00f"),
                                     ("ATTENTION_PRODUCER_REGS", "120"),
                                     ("ATTENTION_CONSUMER_REGS", "184"))),
    Candidate("dep_01f_p120_c184", (("ATTENTION_DEP_SWEEP", "1"),
                                     ("ATTENTION_DEP_MASK", "0x01f"),
                                     ("ATTENTION_PRODUCER_REGS", "120"),
                                     ("ATTENTION_CONSUMER_REGS", "184"))),
    Candidate("reg_p120_c176", (("ATTENTION_PRODUCER_REGS", "120"),
                                 ("ATTENTION_CONSUMER_REGS", "176"))),
    Candidate("reg_p120_c192", (("ATTENTION_PRODUCER_REGS", "120"),
                                 ("ATTENTION_CONSUMER_REGS", "192"))),
    Candidate("reg_p128_c184", (("ATTENTION_PRODUCER_REGS", "128"),
                                 ("ATTENTION_CONSUMER_REGS", "184"))),
    Candidate("single_p128_c200", (("ATTENTION_SINGLE_PIPE0", "1"),
                                    ("ATTENTION_PRODUCER_REGS", "128"),
                                    ("ATTENTION_CONSUMER_REGS", "200"))),
    Candidate("single_p112_c192", (("ATTENTION_SINGLE_PIPE0", "1"),
                                    ("ATTENTION_PRODUCER_REGS", "112"),
                                    ("ATTENTION_CONSUMER_REGS", "192"))),
    Candidate("single_p120_c192", (("ATTENTION_SINGLE_PIPE0", "1"),
                                    ("ATTENTION_PRODUCER_REGS", "120"),
                                    ("ATTENTION_CONSUMER_REGS", "192"))),
    Candidate("single_p136_c200", (("ATTENTION_SINGLE_PIPE0", "1"),
                                    ("ATTENTION_PRODUCER_REGS", "136"),
                                    ("ATTENTION_CONSUMER_REGS", "200"))),
    Candidate("single_p120_c208", (("ATTENTION_SINGLE_PIPE0", "1"),
                                    ("ATTENTION_PRODUCER_REGS", "120"),
                                    ("ATTENTION_CONSUMER_REGS", "208"))),
    Candidate("single_vdbuf_p128_c200", (("ATTENTION_SINGLE_PIPE0", "1"),
                                          ("ATTENTION_SINGLE_PIPE0_V_DOUBLE_BUFFER", "1"),
                                          ("ATTENTION_PRODUCER_REGS", "128"),
                                          ("ATTENTION_CONSUMER_REGS", "200"))),
    Candidate("single_ldx128_p112_c192", (("ATTENTION_SINGLE_PIPE0", "1"),
                                           ("ATTENTION_SINGLE_PIPE0_LD_X128", "1"),
                                           ("ATTENTION_PRODUCER_REGS", "112"),
                                           ("ATTENTION_CONSUMER_REGS", "192"))),
    Candidate("single_ldx128_nosum_p112_c192", (("ATTENTION_SINGLE_PIPE0", "1"),
                                                 ("ATTENTION_SINGLE_PIPE0_LD_X128", "1"),
                                                 ("ATTENTION_SINGLE_PIPE0_LD_X128_SKIP_ROWSUM", "1"),
                                                 ("ATTENTION_PRODUCER_REGS", "112"),
                                                 ("ATTENTION_CONSUMER_REGS", "192"))),
    Candidate("single_pvonce_p112_c192", (("ATTENTION_SINGLE_PIPE0", "1"),
                                           ("ATTENTION_SINGLE_PIPE0_PV_ONCE", "1"),
                                           ("ATTENTION_PRODUCER_REGS", "112"),
                                           ("ATTENTION_CONSUMER_REGS", "192"))),
    Candidate("single_pvh0done_p112_c192", (("ATTENTION_SINGLE_PIPE0", "1"),
                                             ("ATTENTION_SINGLE_PIPE0_PV_H0_DONE", "1"),
                                             ("ATTENTION_PRODUCER_REGS", "112"),
                                             ("ATTENTION_CONSUMER_REGS", "192"))),
    Candidate("single_earlyh1_p112_c192", (("ATTENTION_SINGLE_PIPE0", "1"),
                                            ("ATTENTION_SINGLE_PIPE0_EARLY_P_LD_H1", "1"),
                                            ("ATTENTION_PRODUCER_REGS", "112"),
                                            ("ATTENTION_CONSUMER_REGS", "192"))),
]


def run(cmd, timeout, cwd=ROOT):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def parse_benchmark_csv(path):
    if not path.exists():
        return {"status": "missing_csv", "tflops": ""}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return {"status": "empty_csv", "tflops": ""}
    parts = lines[-1].split(",")
    if len(parts) < 6:
        return {"status": "bad_csv", "tflops": ""}
    return {
        "status": parts[-3],
        "cuda_error": parts[-2],
        "notes": parts[-1],
        "tflops": parts[-4],
        "elapsed_ms": parts[-23] if len(parts) >= 23 else "",
    }


def parse_validation_csv(path):
    if not path.exists():
        return "missing_csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "empty_csv"
    return "ok" if all(row.get("status") == "ok" for row in rows) else "fail"


def parse_ptxas(log):
    kernel_seen = False
    stack = spills_st = spills_ld = regs = ""
    for line in log.splitlines():
        if "qk_tma_mma_ld_kernel" in line and "Function properties" in line:
            kernel_seen = True
            continue
        if kernel_seen and "bytes stack frame" in line:
            m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if m:
                stack, spills_st, spills_ld = m.groups()
            continue
        if kernel_seen and "ptxas info" in line and "Used" in line and "registers" in line:
            m = re.search(r"Used (\d+) registers", line)
            if m:
                regs = m.group(1)
            break
    return regs, stack, spills_st, spills_ld


def compile_candidate(candidate, build_dir, compile_timeout):
    bin_path = build_dir / BIN_REL
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    flags = list(BASE_FLAGS)
    for name, value in candidate.macros:
        flags.append(f"-D{name}={value}")
    cmd = [
        "/usr/local/cuda/bin/nvcc",
        *flags,
        str(SRC.relative_to(ROOT)),
        "-o",
        str(bin_path.relative_to(ROOT)),
        "-lcuda",
    ]
    rc, log = run(cmd, compile_timeout)
    build_log = OUT_DIR / f"attention_1650_{candidate.tag}_build.log"
    build_log.write_text(log)
    regs, stack, spills_st, spills_ld = parse_ptxas(log)
    return rc, bin_path, build_log, regs, stack, spills_st, spills_ld


def perf_candidate(candidate, bin_path, suffix, run_timeout):
    csv_path = OUT_DIR / f"attention_1650_{candidate.tag}{suffix}.csv"
    cmd = [str(bin_path), *PERF_ARGS, "--csv", str(csv_path.relative_to(ROOT))]
    for attempt in range(3):
        try:
            rc, log = run(cmd, run_timeout)
        except subprocess.TimeoutExpired:
            return {"run_rc": "timeout", "status": "timeout", "tflops": "", "csv": str(csv_path)}
        if "busy or unavailable" not in log:
            break
        time.sleep(2.0 + attempt)
    run_log = OUT_DIR / f"attention_1650_{candidate.tag}{suffix}_run.log"
    run_log.write_text(log)
    parsed = parse_benchmark_csv(csv_path)
    parsed["run_rc"] = str(rc)
    parsed["csv"] = str(csv_path)
    return parsed


def validate_candidate(candidate, bin_path, run_timeout):
    csv_path = OUT_DIR / f"attention_1650_{candidate.tag}_validate.csv"
    cmd = [str(bin_path), "--validate-suite", "--csv", str(csv_path.relative_to(ROOT))]
    try:
        rc, log = run(cmd, run_timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    run_log = OUT_DIR / f"attention_1650_{candidate.tag}_validate_run.log"
    run_log.write_text(log)
    if rc != 0:
        return f"rc_{rc}"
    return parse_validation_csv(csv_path)


def candidate_by_tag(tag):
    for cand in CANDIDATES:
        if cand.tag == tag:
            return cand
    raise SystemExit(f"unknown tag: {tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", help="candidate tags to run")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", default=str(OUT_DIR / "attention_1650_summary.csv"))
    parser.add_argument("--rerun-threshold", type=float, default=1594.0)
    parser.add_argument("--validate-threshold", type=float, default=1596.0)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--run-timeout", type=int, default=30)
    args = parser.parse_args()

    if args.list:
        for cand in CANDIDATES:
            macros = " ".join(f"-D{k}={v}" for k, v in cand.macros) or "(defaults)"
            print(f"{cand.tag}: {macros}")
        return

    candidates = [candidate_by_tag(tag) for tag in args.tags] if args.tags else CANDIDATES
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "macros",
                "compile_rc",
                "run_rc",
                "status",
                "cuda_error",
                "tflops",
                "median_tflops",
                "validate",
                "regs",
                "stack",
                "spill_stores",
                "spill_loads",
                "csv",
                "build_log",
                "notes",
            ],
        )
        if write_header:
            writer.writeheader()
        for cand in candidates:
            build_dir = ROOT / f"build_attention_1650_{cand.tag}"
            print(f"[compile] {cand.tag}", flush=True)
            rc, bin_path, build_log, regs, stack, spills_st, spills_ld = compile_candidate(
                cand, build_dir, args.compile_timeout
            )
            macros = " ".join(f"-D{k}={v}" for k, v in cand.macros)
            if rc != 0:
                writer.writerow({
                    "tag": cand.tag,
                    "macros": macros,
                    "compile_rc": rc,
                    "status": "compile_failed",
                    "build_log": str(build_log),
                })
                f.flush()
                continue

            print(f"[perf] {cand.tag}", flush=True)
            perf = perf_candidate(cand, bin_path, "", args.run_timeout)
            values = []
            try:
                first_tflops = float(perf.get("tflops", ""))
            except ValueError:
                first_tflops = 0.0
            if perf.get("status") == "ok":
                values.append(first_tflops)
            if first_tflops >= args.rerun_threshold:
                for i in range(3):
                    print(f"[rerun {i}] {cand.tag}", flush=True)
                    extra = perf_candidate(cand, bin_path, f"_rerun_{i}", args.run_timeout)
                    try:
                        val = float(extra.get("tflops", ""))
                    except ValueError:
                        val = 0.0
                    if extra.get("status") == "ok":
                        values.append(val)
            median = statistics.median(values) if values else ""
            validate = ""
            if isinstance(median, float) and median >= args.validate_threshold:
                print(f"[validate] {cand.tag}", flush=True)
                validate = validate_candidate(cand, bin_path, max(args.run_timeout, 120))
            writer.writerow({
                "tag": cand.tag,
                "macros": macros,
                "compile_rc": rc,
                "run_rc": perf.get("run_rc", ""),
                "status": perf.get("status", ""),
                "cuda_error": perf.get("cuda_error", ""),
                "tflops": perf.get("tflops", ""),
                "median_tflops": f"{median:.3f}" if isinstance(median, float) else "",
                "validate": validate,
                "regs": regs,
                "stack": stack,
                "spill_stores": spills_st,
                "spill_loads": spills_ld,
                "csv": perf.get("csv", ""),
                "build_log": str(build_log),
                "notes": perf.get("notes", ""),
            })
            f.flush()


if __name__ == "__main__":
    try:
        main()
    except subprocess.TimeoutExpired as exc:
        print(f"timeout: {exc}", file=sys.stderr)
        raise SystemExit(124)
