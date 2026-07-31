#!/usr/bin/env python3
"""Generate the audited two-slot alternating K64/K128 N-split ablation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "c0395009e00fcdd2f0a5266ee3cc79e308b51e35e7510ad671683ec7dcd67107"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_between(
    text: str, start: str, end: str, replacement: str, label: str
) -> str:
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise RuntimeError(
            f"{label}: start count {start_count}, end count {end_count}"
        )
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def generate(source: str) -> str:
    text = source

    constants_start = """static constexpr int kStageK = 64;
static constexpr int kStages = 3;
"""
    constants_end = """static constexpr int kCStoreChunkM = 128;
"""
    constants_new = """static constexpr int kStageK = 64; // K64 accounting unit.
static constexpr int kStages = 2;
static constexpr int kShortStageK = 64;
static constexpr int kLongStageK = 128;
static constexpr int kMmaM = 128;
static constexpr int kMmaN = 128;
static constexpr int kMmaK = 16;
static constexpr int kPipes = 2;
static constexpr int kMBlocks = 2;
static constexpr int kBTmaN = 128;
static constexpr int kBTmaNSubtiles = kBTmaN / 64;
static constexpr int kPersistentMacroM = 8;
static constexpr int kPersistentMacroN = 16;

template <int StageK>
struct StageLayout {
  static constexpr int kAWords = kCtaM * StageK / 2;
  static constexpr int kBWords = StageK * kCtaN / 2;
  static constexpr int kBPipeWords = StageK * kMmaN / 2;
  static constexpr int kWords = kAWords + kBWords;
  static constexpr int kABytes = kAWords * static_cast<int>(sizeof(uint32_t));
  static constexpr int kBPipeBytes =
      kBPipeWords * static_cast<int>(sizeof(uint32_t));
  static constexpr int kBytes = kWords * static_cast<int>(sizeof(uint32_t));
};

static constexpr int kShortStageWords = StageLayout<kShortStageK>::kWords;
static constexpr int kLongStageWords = StageLayout<kLongStageK>::kWords;
static constexpr int kMainloopSmemBytes =
    StageLayout<kShortStageK>::kBytes + StageLayout<kLongStageK>::kBytes;

// Max-stage aliases preserve the compile-time layout assertions below.
static constexpr int kAStageWords = StageLayout<kLongStageK>::kAWords;
static constexpr int kBStageWords = StageLayout<kLongStageK>::kBWords;
static constexpr int kBPipeWords = StageLayout<kLongStageK>::kBPipeWords;
static constexpr int kStageWords = StageLayout<kLongStageK>::kWords;
static constexpr int kAStageBytes = StageLayout<kLongStageK>::kABytes;
static constexpr int kBPipeBytes = StageLayout<kLongStageK>::kBPipeBytes;
static constexpr int kStageBytes = StageLayout<kLongStageK>::kBytes;
"""
    text = replace_between(
        text, constants_start, constants_end, constants_new, "stage constants"
    )

    descriptor_start = """__device__ __forceinline__ uint64_t make_stage_a_smem_desc"""
    descriptor_end = """__host__ __device__ __forceinline__ uint32_t make_bf16_idesc()"""
    descriptor_new = """template <int StageK>
__device__ __forceinline__ uint64_t make_stage_a_smem_desc(
    uint32_t *a_smem, int mblock, int mma) {
  // A K128 is physically two adjacent, independently SW128-swizzled K64
  // slabs. This preserves the audited K64 TMA/MMA layout.
  constexpr int kMmasPerK64 = kShortStageK / kMmaK;
  const int k64_slab = mma / kMmasPerK64;
  const int mma_in_slab = mma - k64_slab * kMmasPerK64;
  constexpr int kK64SlabWords = kCtaM * kShortStageK / 2;
  constexpr int kMBlockWords = kMmaM * kShortStageK / 2;
  uint32_t *matrix =
      a_smem + k64_slab * kK64SlabWords + mblock * kMBlockWords;
  return make_sw128_major_k_smem_desc(smem_ptr_u32(matrix), mma_in_slab);
}

__host__ __device__ __forceinline__ uint64_t
make_sw128_major_mn_smem_desc(uint32_t matrix_start_addr, int mma) {
  constexpr uint64_t desc_base =
      (static_cast<uint64_t>(128u) << 16) | (static_cast<uint64_t>(64u) << 32) |
      (static_cast<uint64_t>(1u) << 46) | (static_cast<uint64_t>(2u) << 61);
  constexpr uint32_t kSliceBytes =
      static_cast<uint32_t>(kMmaK * kMmaN / 2 * sizeof(uint32_t));
  const uint32_t addr16 = ((matrix_start_addr & ~0xFu) >> 4) +
                          static_cast<uint32_t>(mma) * (kSliceBytes >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

"""
    text = replace_between(
        text,
        descriptor_start,
        descriptor_end,
        descriptor_new,
        "variable-K descriptors",
    )

    kernel_start = """__device__ __forceinline__ void issue_a_stage_tma"""
    kernel_end = """void encode_a_row_major_sw128_tma_map"""
    kernel_new = r"""template <int StageK>
__device__ __forceinline__ void issue_a_stage_tma(
    const CUtensorMap *a_map, uint32_t *a_smem, uint64_t *ready,
    uint64_t *ready_long, int stage, int tile_m, int k64_cursor) {
  const int a_row = tile_m * kCtaM;
  const int a_col_words = k64_cursor * (kShortStageK / 2);
  constexpr int kK64SlabBytes = kCtaM * kShortStageK * sizeof(uint16_t);
  mbarrier_expect_tx(ready, kK64SlabBytes);
  tma_load_2d(a_map, smem_ptr_u32(a_smem), ready, a_col_words, a_row);
  if constexpr (StageK == kLongStageK) {
    constexpr int kK64SlabWords = kCtaM * kShortStageK / 2;
    mbarrier_expect_tx(ready_long, kK64SlabBytes);
    tma_load_2d(a_map, smem_ptr_u32(a_smem + kK64SlabWords), ready_long,
                a_col_words + kShortStageK / 2, a_row);
  } else if (stage == 1) {
    // Keep the long-slot barrier epoch aligned for a K64 tail. The duplicate
    // slab is unused by MMA.
    constexpr int kK64SlabWords = kCtaM * kShortStageK / 2;
    mbarrier_expect_tx(ready_long, kK64SlabBytes);
    tma_load_2d(a_map, smem_ptr_u32(a_smem + kK64SlabWords), ready_long,
                a_col_words, a_row);
  }
}

template <int StageK>
__device__ __forceinline__ void issue_b_pipe_stage_tma(
    const CUtensorMap *b_map, uint32_t *b_smem, uint64_t *ready, int tile_n,
    int k64_cursor, int pipe) {
  mbarrier_expect_tx(ready, StageLayout<StageK>::kBPipeBytes);
  const int b_col_words = tile_n * (kCtaN / 2) + pipe * (kMmaN / 2);
  const int b_k16 = k64_cursor * (kShortStageK / kMmaK);
  tma_load_4d(b_map, smem_ptr_u32(b_smem), ready, b_col_words, 0, 0, b_k16);
}

template <int StageK>
__device__ __forceinline__ void consume_pipe_stage(
    uint32_t tmem_base, uint32_t idesc, uint32_t *a_smem,
    uint32_t *b_smem, uint64_t *a_ready, uint64_t *b_ready,
    uint64_t *a_ready_long, uint64_t *mma_done, uint64_t *mma_done_long,
    uint32_t tma_phase, int stage, int pipe, int k64_cursor) {
  mbarrier_wait(b_ready, tma_phase);
  mbarrier_wait(a_ready, tma_phase);
  if (stage == 1)
    mbarrier_wait(a_ready_long, tma_phase);
#pragma unroll
  for (int kk = 0; kk < StageK / kMmaK; ++kk) {
    const uint64_t b_desc =
        make_sw128_major_mn_smem_desc(smem_ptr_u32(b_smem), kk);
    const bool input_d = (k64_cursor != 0) || (kk != 0);
#pragma unroll
    for (int mblock = 0; mblock < kMBlocks; ++mblock) {
      const uint64_t a_desc =
          make_stage_a_smem_desc<StageK>(a_smem, mblock, kk);
      const int c_tile = mblock * 2 + pipe;
      tcgen05_mma_bf16_ss(tmem_base + c_tile * kTmemTileStride, a_desc,
                          b_desc, idesc, input_d);
    }
    if constexpr (StageK == kLongStageK) {
      if (kk == kShortStageK / kMmaK - 1)
        tcgen05_commit(mma_done);
    }
  }
  if constexpr (StageK == kLongStageK) {
    tcgen05_commit(mma_done_long);
  } else {
    tcgen05_commit(mma_done);
    if (stage == 1)
      tcgen05_commit(mma_done_long);
  }
}

__device__ __forceinline__ uint32_t *alternating_stage_smem(
    uint32_t *smem, int stage) {
  return smem + (stage == 0 ? 0 : kShortStageWords);
}

template <int ktiles, int mtile_count, int ntile_count>
__global__ __launch_bounds__(kThreads, 1) void gemm256_bf16_16k_kernel(
    const __grid_constant__ CUtensorMap a64_map,
    const __grid_constant__ CUtensorMap a128_map,
    const __grid_constant__ CUtensorMap b64_map,
    const __grid_constant__ CUtensorMap b128_map,
    const __grid_constant__ CUtensorMap c_map, uint32_t *__restrict__ sink) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a64_map;
  (void)a128_map;
  (void)b64_map;
  (void)b128_map;
  (void)c_map;
  (void)sink;
  (void)ktiles;
  (void)mtile_count;
  (void)ntile_count;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr = (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) &
                              ~static_cast<uintptr_t>(1023u);
  uint32_t *smem = reinterpret_cast<uint32_t *>(smem_addr);
  uint32_t *c_store_smem = smem;

  __shared__ uint64_t a_ready[kStages];
  __shared__ uint64_t a_ready_long;
  __shared__ uint64_t b_ready[kPipes][kStages];
  __shared__ uint64_t mma_done[kPipes][kStages];
  __shared__ uint64_t mma_done_long[kPipes];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbarrier_init(&a_ready[s], 1);
#pragma unroll
      for (int p = 0; p < kPipes; ++p)
        mbarrier_init(&b_ready[p][s], 1);
    }
    mbarrier_init(&a_ready_long, 1);
#pragma unroll
    for (int p = 0; p < kPipes; ++p) {
#pragma unroll
      for (int s = 0; s < kStages; ++s)
        mbarrier_init(&mma_done[p][s], 1);
      mbarrier_init(&mma_done_long[p], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane0)
      tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t idesc = make_bf16_idesc() | (1u << 16);
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
  int stage_epoch_base = 0;
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

    int next_stage_epoch = stage_epoch_base;
    int count_cursor = 0;
    while (count_cursor < ktiles) {
      const int stage = next_stage_epoch & 1;
      const int remaining = ktiles - count_cursor;
      count_cursor += (stage == 1 && remaining >= 2) ? 2 : 1;
      ++next_stage_epoch;
    }

    if (warp_id == 0 && lane0) {
      int k64_cursor = 0;
      for (int stage_epoch = stage_epoch_base; k64_cursor < ktiles;
           ++stage_epoch) {
        const int stage = stage_epoch & 1;
        const int remaining = ktiles - k64_cursor;
        const bool use_k128 = stage == 1 && remaining >= 2;
        uint32_t *stage_smem = alternating_stage_smem(smem, stage);
        uint32_t *a_smem = stage_smem;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
#pragma unroll
          for (int p = 0; p < kPipes; ++p) {
            mbarrier_wait(&mma_done[p][stage], reuse_phase);
            if (stage == 1)
              mbarrier_wait(&mma_done_long[p], reuse_phase);
          }
        }
        if (use_k128) {
          uint32_t *b_smem =
              stage_smem + StageLayout<kLongStageK>::kAWords;
          issue_a_stage_tma<kLongStageK>(
              &a128_map, a_smem, &a_ready[stage], &a_ready_long, stage, tile_m,
              k64_cursor);
          issue_b_pipe_stage_tma<kLongStageK>(
              &b128_map, b_smem, &b_ready[0][stage], tile_n, k64_cursor, 0);
          k64_cursor += 2;
        } else {
          uint32_t *b_smem =
              stage_smem +
              (stage == 1 ? StageLayout<kLongStageK>::kAWords
                          : StageLayout<kShortStageK>::kAWords);
          issue_a_stage_tma<kShortStageK>(
              &a64_map, a_smem, &a_ready[stage], &a_ready_long, stage, tile_m,
              k64_cursor);
          issue_b_pipe_stage_tma<kShortStageK>(
              &b64_map, b_smem, &b_ready[0][stage], tile_n, k64_cursor, 0);
          ++k64_cursor;
        }
      }
    }

    if (warp_id == 1 && lane0) {
      int k64_cursor = 0;
      for (int stage_epoch = stage_epoch_base; k64_cursor < ktiles;
           ++stage_epoch) {
        const int stage = stage_epoch & 1;
        const int remaining = ktiles - k64_cursor;
        const bool use_k128 = stage == 1 && remaining >= 2;
        uint32_t *stage_smem = alternating_stage_smem(smem, stage);
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase = static_cast<uint32_t>(
              ((stage_epoch - kStages) / kStages) & 1);
          mbarrier_wait(&mma_done[1][stage], reuse_phase);
          if (stage == 1)
            mbarrier_wait(&mma_done_long[1], reuse_phase);
        }
        if (use_k128) {
          uint32_t *b_smem =
              stage_smem + StageLayout<kLongStageK>::kAWords +
              StageLayout<kLongStageK>::kBPipeWords;
          issue_b_pipe_stage_tma<kLongStageK>(
              &b128_map, b_smem, &b_ready[1][stage], tile_n, k64_cursor, 1);
          k64_cursor += 2;
        } else {
          uint32_t *b_smem =
              stage_smem +
              (stage == 1 ? StageLayout<kLongStageK>::kAWords +
                                StageLayout<kLongStageK>::kBPipeWords
                          : StageLayout<kShortStageK>::kAWords +
                                StageLayout<kShortStageK>::kBPipeWords);
          issue_b_pipe_stage_tma<kShortStageK>(
              &b64_map, b_smem, &b_ready[1][stage], tile_n, k64_cursor, 1);
          ++k64_cursor;
        }
      }
    }

    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
      int k64_cursor = 0;
      int last_stage_epoch = stage_epoch_base;
      for (int stage_epoch = stage_epoch_base; k64_cursor < ktiles;
           ++stage_epoch) {
        const int stage = stage_epoch & 1;
        const int remaining = ktiles - k64_cursor;
        const bool use_k128 = stage == 1 && remaining >= 2;
        const uint32_t tma_phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        uint32_t *stage_smem = alternating_stage_smem(smem, stage);
        uint32_t *a_smem = stage_smem;
        if (use_k128) {
          uint32_t *b_smem =
              stage_smem + StageLayout<kLongStageK>::kAWords +
              pipe * StageLayout<kLongStageK>::kBPipeWords;
          consume_pipe_stage<kLongStageK>(
              tmem_base, idesc, a_smem, b_smem, &a_ready[stage],
              &b_ready[pipe][stage], &a_ready_long, &mma_done[pipe][stage],
              &mma_done_long[pipe], tma_phase, stage, pipe, k64_cursor);
          k64_cursor += 2;
        } else {
          uint32_t *b_smem =
              stage_smem +
              (stage == 1 ? StageLayout<kLongStageK>::kAWords +
                                pipe * StageLayout<kLongStageK>::kBPipeWords
                          : StageLayout<kShortStageK>::kAWords +
                                pipe * StageLayout<kShortStageK>::kBPipeWords);
          consume_pipe_stage<kShortStageK>(
              tmem_base, idesc, a_smem, b_smem, &a_ready[stage],
              &b_ready[pipe][stage], &a_ready_long, &mma_done[pipe][stage],
              &mma_done_long[pipe], tma_phase, stage, pipe, k64_cursor);
          ++k64_cursor;
        }
        last_stage_epoch = stage_epoch;
      }
      // A K128 group contains twice as many MMA issues. Completion groups from
      // the two slots are not assumed to retire in slot order, so make both
      // slots safe before the epilogue reuses the mainloop SMEM.
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        const int delta = (last_stage_epoch - s) & 1;
        const int last_epoch_for_slot = last_stage_epoch - delta;
        if (last_epoch_for_slot >= stage_epoch_base) {
          const uint32_t last_phase = static_cast<uint32_t>(
              (last_epoch_for_slot / kStages) & 1);
          mbarrier_wait(&mma_done[pipe][s], last_phase);
          if (s == 1)
            mbarrier_wait(&mma_done_long[pipe], last_phase);
        }
      }
    }
    __syncthreads();

    const int global_row_base = tile_m * kCtaM;
    const int global_col_base = tile_n * kCtaN;
    store_256x256_float_tile_tma(tmem_base, &c_map, c_store_smem,
                                 global_row_base, global_col_base);
    ++tile_iter;
    stage_epoch_base = next_stage_epoch;
  }

  if (threadIdx.x == 0)
    tcgen05_fence_after_thread_sync();
  __syncthreads();
  if (warp_id == 0)
    tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0)
    tcgen05_relinquish_alloc_permit();
#endif
}

"""
    text = replace_between(
        text,
        kernel_start,
        kernel_end,
        kernel_new,
        "alternating-K TMA and kernel",
    )

    a_encoder_start = """void encode_a_row_major_sw128_tma_map"""
    a_encoder_end = """void encode_b_row_major_sw128_k16_tma_map"""
    a_encoder_new = """void encode_a_row_major_sw128_tma_map(
    CUtensorMap *map, void *base, uint64_t rows, uint64_t cols_bf16,
    int box_stage_k) {
  const cuuint64_t cols_words = cols_bf16 / 2;
  const cuuint64_t global_dim[2] = {cols_words, rows};
  const cuuint64_t global_stride[1] = {cols_words * sizeof(uint32_t)};
  const cuuint32_t box_dim[2] = {kShortStageK / 2, kCtaM};
  const cuuint32_t elem_stride[2] = {1, 1};
  driver_check(cuTensorMapEncodeTiled(
                   map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, base, global_dim,
                   global_stride, box_dim, elem_stride,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(a_row_major_sw128)");
  (void)box_stage_k;
}

"""
    text = replace_between(
        text,
        a_encoder_start,
        a_encoder_end,
        a_encoder_new,
        "A K64 slab map",
    )
    text = text.replace(
        """void encode_b_row_major_sw128_k16_tma_map(CUtensorMap *map, void *base,
                                          uint64_t rows, uint64_t cols_bf16) {""",
        """void encode_b_row_major_sw128_k16_tma_map(
    CUtensorMap *map, void *base, uint64_t rows, uint64_t cols_bf16,
    int box_stage_k) {""",
        1,
    )
    text = replace_once(
        text,
        """  const cuuint32_t box_dim[4] = {kMmaN / 4, kMmaK, kBTmaNSubtiles,
                                 kStageK / kMmaK};""",
        """  const cuuint32_t box_dim[4] = {
      kMmaN / 4, kMmaK, kBTmaNSubtiles,
      static_cast<cuuint32_t>(box_stage_k / kMmaK)};""",
        "B variable box",
    )

    launch_start = """template <int KTiles, int MTiles, int NTiles>
void set_one_gemm_kernel_attribute()"""
    launch_end = """struct CaseResult {"""
    launch_new = r"""template <int KTiles, int MTiles, int NTiles>
void set_one_gemm_kernel_attribute() {
  cuda_check(cudaFuncSetAttribute(
      gemm256_bf16_16k_kernel<KTiles, MTiles, NTiles>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes));
}

void set_gemm_kernel_attribute() {
  set_one_gemm_kernel_attribute<256, 64, 64>();
  set_one_gemm_kernel_attribute<8, 2, 2>();
  set_one_gemm_kernel_attribute<4, 1, 1>();
}

template <int KTiles, int MTiles, int NTiles>
void launch_one_gemm_kernel(
    dim3 grid, const CUtensorMap &a64_map, const CUtensorMap &a128_map,
    const CUtensorMap &b64_map, const CUtensorMap &b128_map,
    const CUtensorMap &c_map, uint32_t *d_sink) {
  gemm256_bf16_16k_kernel<KTiles, MTiles, NTiles>
      <<<grid, kThreads, kDynamicSmemBytes>>>(
          a64_map, a128_map, b64_map, b128_map, c_map, d_sink);
}

void launch_gemm_kernel(
    dim3 grid, const CUtensorMap &a64_map, const CUtensorMap &a128_map,
    const CUtensorMap &b64_map, const CUtensorMap &b128_map,
    const CUtensorMap &c_map, uint32_t *d_sink, int ktiles, int mtile,
    int ntile) {
  if (ktiles == 256 && mtile == 64 && ntile == 64) {
    launch_one_gemm_kernel<256, 64, 64>(
        grid, a64_map, a128_map, b64_map, b128_map, c_map, d_sink);
  } else if (ktiles == 8 && mtile == 2 && ntile == 2) {
    launch_one_gemm_kernel<8, 2, 2>(
        grid, a64_map, a128_map, b64_map, b128_map, c_map, d_sink);
  } else if (ktiles == 4 && mtile == 1 && ntile == 1) {
    launch_one_gemm_kernel<4, 1, 1>(
        grid, a64_map, a128_map, b64_map, b128_map, c_map, d_sink);
  } else {
    std::fprintf(stderr, "Unsupported specialized shape: %d/%d/%d\n",
                 ktiles, mtile, ntile);
    std::exit(EXIT_FAILURE);
  }
}
"""
    text = replace_between(
        text, launch_start, launch_end, launch_new, "alternating-K launch"
    )

    map_old = """  CUtensorMap a_map{}, b_map{}, c_map{};
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n);
  encode_c_row_major_float_tma_map(&c_map, d_c, m, n);
"""
    map_new = """  CUtensorMap a64_map{}, a128_map{}, b64_map{}, b128_map{}, c_map{};
  encode_a_row_major_sw128_tma_map(
      &a64_map, d_a, m, k, kShortStageK);
  encode_a_row_major_sw128_tma_map(
      &a128_map, d_a, m, k, kLongStageK);
  encode_b_row_major_sw128_k16_tma_map(
      &b64_map, d_b, k, n, kShortStageK);
  encode_b_row_major_sw128_k16_tma_map(
      &b128_map, d_b, k, n, kLongStageK);
  encode_c_row_major_float_tma_map(&c_map, d_c, m, n);
"""
    if text.count(map_old) != 2:
        raise RuntimeError(f"host tensor maps: expected two anchors, found {text.count(map_old)}")
    text = text.replace(map_old, map_new)

    launch_call_old = """    launch_gemm_kernel(grid, a_map, b_map, c_map, d_sink, ktiles, mtile, ntile);"""
    launch_call_new = """    launch_gemm_kernel(
        grid, a64_map, a128_map, b64_map, b128_map, c_map, d_sink, ktiles,
        mtile, ntile);"""
    text = replace_once(
        text, launch_call_old, launch_call_new, "performance launch call"
    )
    validation_call_old = """  launch_gemm_kernel(grid, a_map, b_map, c_map, d_sink, ktiles, mtile, ntile);"""
    validation_call_new = """  launch_gemm_kernel(
      grid, a64_map, a128_map, b64_map, b128_map, c_map, d_sink, ktiles,
      mtile, ntile);"""
    text = replace_once(
        text, validation_call_old, validation_call_new, "validation launch call"
    )

    text = replace_once(
        text,
        '"stages=3 pipes=2 persistent_ctas=%d scheduler=static_%dx%d_mfast ',
        '"stages=2 alternating_k=64/128 pipes=2 persistent_ctas=%d '
        'scheduler=static_%dx%d_mfast ',
        "configuration banner",
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent
        / "baseline"
        / "gemm256_bf16_16k.cu",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "refusing unaudited source: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )
    generated = generate(source_bytes.decode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    print(
        f"source_sha256={source_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
