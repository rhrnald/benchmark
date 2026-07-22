#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm_nonmulticast_l2_evict}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
definition_commit=${DEFINITION_COMMIT:?set DEFINITION_COMMIT to the checked-out definition commit}
expected_kernel_sha256=8171bbac9aebbe4dce866feb598de16f52ad76275d63d5d303721f098ce689a6

if [[ -e "$out_dir" ]]; then
  echo "refusing to reuse existing output directory: $out_dir" >&2
  exit 2
fi
mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
exec > >(tee "$out_dir/driver.log") 2>&1
cd "$src_dir"

read -r actual_kernel_sha256 _ < <(sha256sum gemm256_tma_tcgen05_bench.cu)
if [[ "$actual_kernel_sha256" != "$expected_kernel_sha256" ]]; then
  echo "unexpected kernel source SHA-256: $actual_kernel_sha256" >&2
  exit 3
fi

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_A_INPUT=0 -DGEMM_REPEAT_B_INPUT=0
  -DGEMM_REPEAT_B_BROADCAST=0 -DGEMM_REPEAT_TUNING=1
  -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_TMA_MULTICAST_A=0 -DGEMM_TMA_MULTICAST_B=0
  -DGEMM_PERSISTENT_CTA=1 -DGEMM_PERSISTENT_STATIC_SCHEDULER=0
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16
  -DGEMM_PERSISTENT_32K_MACRO_M=16 -DGEMM_PERSISTENT_32K_MACRO_N=16
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1
  -DGEMM_TMA_A_L2_PROMOTION=CU_TENSOR_MAP_L2_PROMOTION_NONE
  -DGEMM_TMA_B_L2_PROMOTION=CU_TENSOR_MAP_L2_PROMOTION_NONE
  -DGEMM_TMA_C_L2_PROMOTION=CU_TENSOR_MAP_L2_PROMOTION_NONE
)

: > "$out_dir/compile_commands.txt"
compile_variant() {
  local name=$1
  local a_policy=$2
  local b_policy=$3
  local command=(
    "$nvcc_bin" "${common[@]}"
    "-DGEMM_TMA_A_L2_EVICT_POLICY=$a_policy"
    "-DGEMM_TMA_B_L2_EVICT_POLICY=$b_policy"
    gemm256_tma_tcgen05_bench.cu -lcuda -o "$out_dir/bin/$name"
  )
  printf '%q ' "${command[@]}" >> "$out_dir/compile_commands.txt"
  printf '\n' >> "$out_dir/compile_commands.txt"
  "${command[@]}"
}

# Policy encoding: 0=no hint, 1=evict_last, 2=evict_first.
compile_variant baseline 0 0
compile_variant a_last 1 0
compile_variant b_last 0 1
compile_variant a_last_b_first 1 2
compile_variant a_first_b_last 2 1

python3 test_persistent_l2_mapping.py > "$out_dir/mapping_test.log"
cp gemm256_tma_tcgen05_bench.cu test_persistent_l2_mapping.py \
  NON_MULTICAST_L2_EXPERIMENT_PLAN.md "$out_dir/source/"
cp ../run_b200_gemm_nonmulticast_l2_evict.sh "$out_dir/source/"
printf '%s\n' "$definition_commit" > "$out_dir/definition_commit.txt"
printf '%s\n' "$expected_kernel_sha256" \
  > "$out_dir/kernel_expected_sha256.txt"
printf '%s\n' \
  '0=no_hint' '1=L2::evict_last' '2=L2::evict_first' \
  > "$out_dir/policy_encoding.txt"
"$nvcc_bin" --version > "$out_dir/nvcc_version.txt"

variants=(baseline a_last b_last a_last_b_first a_first_b_last)
for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 148 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"
printf 'pass\tposition\tvariant\tcsv\n' > "$out_dir/sequence.tsv"
printf 'pass,timestamp,uuid,temp_c,power_w,sm_clock_mhz,gpu_util_pct\n' \
  > "$out_dir/pass_telemetry.csv"
: > "$out_dir/raw.log"

count=${#variants[@]}
for pass_index in 0 1 2; do
  pass=$((pass_index + 1))
  order=()
  if [[ $pass == 1 ]]; then
    order=("${variants[@]}")
  elif [[ $pass == 2 ]]; then
    for ((i=count-1; i>=0; --i)); do order+=("${variants[$i]}"); done
  else
    for ((i=0; i<count; ++i)); do
      index=$(((i + 2) % count))
      order+=("${variants[$index]}")
    done
  fi

  telemetry=$(nvidia-smi -i 0 \
    --query-gpu=timestamp,uuid,temperature.gpu,power.draw,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits)
  printf '%d,%s\n' "$pass" "$telemetry" >> "$out_dir/pass_telemetry.csv"

  position=0
  for variant in "${order[@]}"; do
    position=$((position + 1))
    csv_name="${variant}_p${pass}.csv"
    printf '%d\t%d\t%s\t%s\n' "$pass" "$position" "$variant" "$csv_name" \
      >> "$out_dir/sequence.tsv"
    echo "pass=$pass position=$position variant=$variant size=16384 ctas=148" \
      | tee -a "$out_dir/raw.log"
    "$out_dir/bin/$variant" --sizes 16384 --warmup 1 --iters 5 \
      --input-init random --persistent-ctas 148 \
      --csv "$out_dir/csv/$csv_name" \
      2>&1 | tee -a "$out_dir/raw.log"
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* \
  "$out_dir"/csv/* "$out_dir"/validate_*.log \
  "$out_dir/compile_commands.txt" "$out_dir/definition_commit.txt" \
  "$out_dir/kernel_expected_sha256.txt" "$out_dir/mapping_test.log" \
  "$out_dir/nvcc_version.txt" "$out_dir/nvidia_smi_before.txt" \
  "$out_dir/nvidia_smi_after.txt" "$out_dir/pass_telemetry.csv" \
  "$out_dir/policy_encoding.txt" "$out_dir/raw.log" \
  "$out_dir/sequence.tsv" > "$out_dir/sha256.txt"
