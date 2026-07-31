#!/usr/bin/env bash
set -euo pipefail

cuda_root=${CUDA_ROOT:-/usr/local/cuda}
nvcc=${NVCC:-${cuda_root}/bin/nvcc}
cuobjdump=${CUOBJDUMP:-${cuda_root}/bin/cuobjdump}
root=${1:-/workspace/gemm_spatial_order}

variants=(direct table_identity morton hilbert hilbert_transpose hilbert_reverse)
inputs=(random random-signed8)

mkdir -p "${root}/bin" "${root}/logs" "${root}/runs/random" \
  "${root}/runs/random-signed8"

nvidia-smi -q > "${root}/logs/nvidia_smi_before.txt"
"${nvcc}" --version > "${root}/logs/nvcc_version.txt"

for variant in "${variants[@]}"; do
  "${nvcc}" -O3 -std=c++17 \
    -gencode=arch=compute_100a,code=sm_100a \
    -Xptxas=-v "${root}/source/${variant}.cu" -lcuda \
    -o "${root}/bin/${variant}" \
    > "${root}/logs/build_${variant}.txt" 2>&1
  "${cuobjdump}" --dump-resource-usage "${root}/bin/${variant}" \
    > "${root}/logs/resources_${variant}.txt"
done

for variant in "${variants[@]}"; do
  for pattern in pattern ones; do
    "${root}/bin/${variant}" --validate --validate-size 512 \
      --validate-pattern "${pattern}" \
      > "${root}/logs/validate_${variant}_${pattern}.txt"
  done
done

# Six cyclic passes put every variant in every execution position once.
# Each process performs one warmup and five timed launches.
for input in "${inputs[@]}"; do
  for pass in 0 1 2 3 4 5; do
    for offset in 0 1 2 3 4 5; do
      index=$(((pass + offset) % 6))
      variant=${variants[$index]}
      output_dir="${root}/runs/${input}"
      "${root}/bin/${variant}" --warmup 1 --iters 5 \
        --input-init "${input}" \
        --csv "${output_dir}/${variant}_pass$((pass + 1)).csv" \
        > "${output_dir}/${variant}_pass$((pass + 1)).txt"
    done
  done
done

nvidia-smi -q > "${root}/logs/nvidia_smi_after.txt"
