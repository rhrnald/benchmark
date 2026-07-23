#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=${BIN_DIR:-/workspace/e7a_cutlass_16k_compare_20260723/bin}
OUT_DIR=${OUT_DIR:-/workspace/e7a_cutlass_16k_compare_20260723/results}

E7A_BIN="$BIN_DIR/gemm_e7a_clean"
CUTLASS_BIN="$BIN_DIR/cutlass_bf16_best_clc_bench"

mkdir -p "$OUT_DIR/logs"
exec > >(tee "$OUT_DIR/comparison_raw.log") 2>&1

if [[ ! -x "$E7A_BIN" || ! -x "$CUTLASS_BIN" ]]; then
  echo "missing benchmark binary under $BIN_DIR" >&2
  exit 1
fi

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader > "$output"
}

run_e7a() {
  local pass=$1
  echo "BEGIN method=e7a input=unit size=16384 pass=$pass"
  snapshot_gpu "$OUT_DIR/logs/gpu_e7a_p${pass}_before.csv"
  "$E7A_BIN" \
    --warmup 1 --iters 5 --input-init random \
    --csv "$OUT_DIR/e7a_unit_16384_p${pass}.csv" \
    2>&1 | tee "$OUT_DIR/logs/e7a_p${pass}.log"
  echo "END method=e7a input=unit size=16384 pass=$pass"
}

run_cutlass() {
  local pass=$1
  echo "BEGIN method=cutlass input=unit size=16384 pass=$pass"
  snapshot_gpu "$OUT_DIR/logs/gpu_cutlass_p${pass}_before.csv"
  "$CUTLASS_BIN" \
    --m=16384 --n=16384 --k=16384 \
    --warmup=1 --iterations=5 --input-dist=unit \
    2>&1 | tee "$OUT_DIR/logs/cutlass_p${pass}.log"
  echo "END method=cutlass input=unit size=16384 pass=$pass"
}

echo "PROTOCOL one_case_per_process=true warmup=1 timed=5 input=BF16_[0,1)"
snapshot_gpu "$OUT_DIR/logs/gpu_start.csv"
sha256sum "$E7A_BIN" "$CUTLASS_BIN" | tee "$OUT_DIR/binaries.sha256"

"$E7A_BIN" \
  --validate --validate-size 512 --validate-pattern pattern \
  2>&1 | tee "$OUT_DIR/logs/e7a_validate_pattern.log"
"$E7A_BIN" \
  --validate --validate-size 512 --validate-pattern ones \
  2>&1 | tee "$OUT_DIR/logs/e7a_validate_ones.log"

printf "position\tpass\tmethod\n" > "$OUT_DIR/sequence.tsv"

printf "1\t1\te7a\n2\t1\tcutlass\n" >> "$OUT_DIR/sequence.tsv"
run_e7a 1
run_cutlass 1

printf "1\t2\tcutlass\n2\t2\te7a\n" >> "$OUT_DIR/sequence.tsv"
run_cutlass 2
run_e7a 2

printf "1\t3\te7a\n2\t3\tcutlass\n" >> "$OUT_DIR/sequence.tsv"
run_e7a 3
run_cutlass 3

printf "1\t4\tcutlass\n2\t4\te7a\n" >> "$OUT_DIR/sequence.tsv"
run_cutlass 4
run_e7a 4

snapshot_gpu "$OUT_DIR/logs/gpu_end.csv"
