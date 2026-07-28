#!/usr/bin/env python3
import argparse
from pathlib import Path


KERNEL = r"""template <int ktiles, int mtile_count, int ntile_count>
__global__ __launch_bounds__(kThreads, 1) void tma_load_only_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap c_map, uint32_t *__restrict__ sink) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a_map;
  (void)b_map;
  (void)c_map;
  (void)sink;
  (void)ktiles;
  (void)mtile_count;
  (void)ntile_count;
#else
  (void)c_map;
  (void)sink;
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr = (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) &
                              ~static_cast<uintptr_t>(1023u);
  uint32_t *smem = reinterpret_cast<uint32_t *>(smem_addr);

  __shared__ uint64_t a_ready[kStages];
  __shared__ uint64_t b_ready[kPipes][kStages];
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbarrier_init(&a_ready[s], 1);
#pragma unroll
      for (int p = 0; p < kPipes; ++p)
        mbarrier_init(&b_ready[p][s], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;

  const int persistent_macro_m =
      mtile_count < kPersistentMacroM ? mtile_count : kPersistentMacroM;
  const int persistent_macro_n =
      ntile_count < kPersistentMacroN ? ntile_count : kPersistentMacroN;
  const int persistent_groups_m =
      (mtile_count + persistent_macro_m - 1) / persistent_macro_m;
  const int persistent_groups_n =
      (ntile_count + persistent_macro_n - 1) / persistent_macro_n;
  const int persistent_macro_tiles = persistent_macro_m * persistent_macro_n;
  const int persistent_task_count =
      persistent_groups_m * persistent_groups_n * persistent_macro_tiles;

  int tile_iter = 0;
  int static_linear_tile = static_cast<int>(blockIdx.x);
  while (true) {
    const int linear_tile = static_linear_tile;
    static_linear_tile += static_cast<int>(gridDim.x);
    if (linear_tile >= persistent_task_count)
      break;

    const int macro_id = linear_tile / persistent_macro_tiles;
    const int local = linear_tile - macro_id * persistent_macro_tiles;
    const int macro_n = macro_id % persistent_groups_n;
    const int macro_m = macro_id / persistent_groups_n;
    const int local_m = local % persistent_macro_m;
    const int local_n = local / persistent_macro_m;
    const int tile_m = macro_m * persistent_macro_m + local_m;
    const int tile_n = macro_n * persistent_macro_n + local_n;
    if (tile_m >= mtile_count || tile_n >= ntile_count) {
      __syncthreads();
      continue;
    }

    const int stage_epoch_base = tile_iter * ktiles;
    if (warp_id == 0 && lane0) {
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords;
        if (stage_epoch >= kStages) {
          const uint32_t prior_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
          mbarrier_wait(&a_ready[stage], prior_phase);
          mbarrier_wait(&b_ready[0][stage], prior_phase);
        }
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
      }
#pragma unroll
      for (int tail = 0; tail < kStages; ++tail) {
        const int stage_epoch = stage_epoch_base + ktiles - kStages + tail;
        const int stage = stage_epoch % kStages;
        const uint32_t phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        mbarrier_wait(&a_ready[stage], phase);
        mbarrier_wait(&b_ready[0][stage], phase);
      }
    }

    if (warp_id == 1 && lane0) {
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *b_smem = stage_smem + kAStageWords + kBPipeWords;
        if (stage_epoch >= kStages) {
          const uint32_t prior_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
          mbarrier_wait(&b_ready[1][stage], prior_phase);
        }
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
      }
#pragma unroll
      for (int tail = 0; tail < kStages; ++tail) {
        const int stage_epoch = stage_epoch_base + ktiles - kStages + tail;
        const int stage = stage_epoch % kStages;
        const uint32_t phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        mbarrier_wait(&b_ready[1][stage], phase);
      }
    }

    __syncthreads();
    ++tile_iter;
  }
#endif
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = Path(args.base).read_text()
    marker = (
        "template <int ktiles, int mtile_count, int ntile_count>\n"
        "__global__ __launch_bounds__(kThreads, 1) void "
        "gemm256_bf16_16k_kernel("
    )
    start = text.index(marker)
    end = text.index("\nvoid encode_a_row_major_sw128_tma_map", start)
    text = text[:start] + KERNEL + text[end:]
    text = text.replace("gemm256_bf16_16k_kernel", "tma_load_only_kernel")
    text = text.replace(
        "scheduler=static_%dx%d_mfast overhead=fixed_sink ",
        "scheduler=static_%dx%d_mfast mode=tma_load_only ",
    )
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
