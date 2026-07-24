#!/usr/bin/env python3
"""Generate the hash-gated N-split cross-tile K64-prefetch ablation.

Both modes claim the next valid persistent task before the current epilogue,
reserve that task's normal kt0 mainloop stage, and rotate the two 64 KiB C
staging buffers over the other physical stages.  Both execute the same two
noinline helper calls, with one disabled by a uniform ``do_issue`` argument.
Only the active call position and the host-visible banner differ:

* nonoverlap: issue next kt0 after both C-store groups have drained;
* overlap: issue next kt0 after C-store group 0 commits and before it drains.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_INPUT_SHA256 = (
    "a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc"
)

MODES = ("nonoverlap", "overlap")

STATIC_ASSERT_ANCHOR = """static_assert(kBTmaN == 128);
"""

PREFETCH_STATIC_ASSERTS = """static_assert(kBTmaN == 128);
static_assert(kStages == 3);
static_assert(kCStoreBuffers == 2);
static_assert(kCStoreChunkCount == 4);
static_assert(kCStoreStageWords == kStageWords);
static_assert(kCStoreStageBytes == kStageBytes);
"""

C_STORE_POINTER = """  uint32_t *c_store_smem = smem;
"""

C_STORE_POINTER_REPLACEMENT = (
    "  // C-store pointers are selected per group after reserving next kt0.\n"
)

STORE_REGION_START = """__device__ __forceinline__ void
store_256x256_float_tile_tma"""

STORE_REGION_END = """__device__ __forceinline__ void issue_a_stage_tma"""

GROUPED_C_STORE = r"""// Issue and commit one pair of 128x128 C chunks.  The caller
// supplies two explicit, disjoint 64 KiB staging pointers.
__device__ __forceinline__ void
issue_float_c_group_tma(uint32_t tmem_base, const CUtensorMap *c_map,
                        uint32_t *c_smem0, uint32_t *c_smem1, int group,
                        int row_offset, int col_offset) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#pragma unroll
  for (int i = 0; i < kCStoreBuffers; ++i) {
    const int chunk = group + i;
    uint32_t *tile_smem = i == 0 ? c_smem0 : c_smem1;
    const int chunk_m = chunk / kCStoreChunksN;
    const int chunk_n = chunk - chunk_m * kCStoreChunksN;
    const int tile_row = row_offset + chunk_m * kCStoreChunkM;
    const int tile_col = col_offset + chunk_n * kCStoreChunkN;
    issue_float_c_chunk_tma(tmem_base, c_map, tile_smem, chunk_m, chunk_n,
                            tile_row, tile_col);
  }
  if (threadIdx.x == 0)
    tma_store_commit_group();
#else
  (void)tmem_base;
  (void)c_map;
  (void)c_smem0;
  (void)c_smem1;
  (void)group;
  (void)row_offset;
  (void)col_offset;
#endif
}

__device__ __forceinline__ void wait_float_c_group_tma() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (threadIdx.x == 0)
    tma_store_wait_group_0();
  __syncthreads();
#endif
}

"""

KERNEL_ANCHOR = """__global__ __launch_bounds__(kThreads, 1) void gemm256_bf16_16k_kernel"""

PREFETCH_HELPERS = r"""__device__ __forceinline__ void decode_persistent_task(
    int linear_tile, int persistent_macro_tiles, int persistent_groups_n,
    int persistent_macro_m, int persistent_macro_n, int &tile_m,
    int &tile_n) {
  const int macro_id = linear_tile / persistent_macro_tiles;
  const int local = linear_tile - macro_id * persistent_macro_tiles;
  const int macro_n = macro_id % persistent_groups_n;
  const int macro_m = macro_id / persistent_groups_n;
  const int local_m = local % persistent_macro_m;
  const int local_n = local / persistent_macro_m;
  tile_m = macro_m * persistent_macro_m + local_m;
  tile_n = macro_n * persistent_macro_n + local_n;
}

// Claim through padded macroblock positions so every published task is either
// a valid output tile or the terminal sentinel.  Padded positions consume no
// mainloop barrier epochs.
__device__ __forceinline__ int claim_next_valid_persistent_task(
    uint32_t *counter, int persistent_task_count,
    int persistent_macro_tiles, int persistent_groups_n,
    int persistent_macro_m, int persistent_macro_n, int mtile_count,
    int ntile_count) {
  while (true) {
    const int linear_tile = static_cast<int>(atomicAdd(counter, 1u));
    if (linear_tile >= persistent_task_count)
      return linear_tile;
    int tile_m;
    int tile_n;
    decode_persistent_task(
        linear_tile, persistent_macro_tiles, persistent_groups_n,
        persistent_macro_m, persistent_macro_n, tile_m, tile_n);
    if (tile_m < mtile_count && tile_n < ntile_count)
      return linear_tile;
  }
}

// Issue the next output tile's normal kt0 transaction into its normal logical
// and physical stage.  The ready-barrier generation remains the unmodified
// global stage epoch.  Waiting for both pipe completions protects the shared A
// panel; each B half is protected by its owning pipe.
__device__ __noinline__ void prefetch_next_tile_kt0(
    bool do_issue, int next_linear_tile, int persistent_task_count,
    int persistent_macro_tiles, int persistent_groups_n,
    int persistent_macro_m, int persistent_macro_n, int tile_iter,
    int ktiles, const CUtensorMap *a_map, const CUtensorMap *b_map,
    uint32_t *smem, uint64_t *a_ready, uint64_t (*b_ready)[kStages],
    uint64_t (*mma_done)[kStages]) {
  if (!do_issue || next_linear_tile >= persistent_task_count)
    return;

  int next_tile_m;
  int next_tile_n;
  decode_persistent_task(
      next_linear_tile, persistent_macro_tiles, persistent_groups_n,
      persistent_macro_m, persistent_macro_n, next_tile_m, next_tile_n);
  const int next_stage_epoch = (tile_iter + 1) * ktiles;
  const int stage = next_stage_epoch % kStages;
  const uint32_t reuse_phase = static_cast<uint32_t>(
      ((next_stage_epoch - kStages) / kStages) & 1);
  uint32_t *stage_smem = smem + stage * kStageWords;
  uint32_t *a_smem = stage_smem;

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;
  if (warp_id == 0 && lane0) {
#pragma unroll
    for (int p = 0; p < kPipes; ++p)
      mbarrier_wait(&mma_done[p][stage], reuse_phase);
    issue_a_stage_tma(a_map, a_smem, &a_ready[stage], next_tile_m, 0);
    issue_b_pipe_stage_tma(
        b_map, stage_smem + kAStageWords, &b_ready[0][stage],
        next_tile_n, 0, 0);
  }
  if (warp_id == 1 && lane0) {
    mbarrier_wait(&mma_done[1][stage], reuse_phase);
    issue_b_pipe_stage_tma(
        b_map, stage_smem + kAStageWords + kBPipeWords,
        &b_ready[1][stage], next_tile_n, 0, 1);
  }
}

"""

LOOP_REGION_START = (
    "  // The dynamic counter hands out one 16x16-macroblock position at a time.\n"
)
LOOP_REGION_END = "  } // persistent output-tile loop\n"

PREFETCH_CALL_TEMPLATE = """    prefetch_next_tile_kt0(
        @DO_ISSUE@, next_linear_tile, persistent_task_count, persistent_macro_tiles,
        persistent_groups_n, persistent_macro_m, persistent_macro_n,
        tile_iter, ktiles, &a_map, &b_map, smem, a_ready, b_ready,
        mma_done);"""

LOOP_TEMPLATE = r"""  // The dynamic counter hands out one 16x16-macroblock position at a time.
  // M varies fastest inside a macroblock so neighboring workers share B;
  // macro N varies fastest so the next macroblock retains the same M range.
  // One lookahead task is reserved by this CTA before its current epilogue.
  const int total_tiles = mtile_count * ntile_count;
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
  uint32_t *persistent_counter = sink + total_tiles;

  if (threadIdx.x == 0) {
    persistent_task_shared = claim_next_valid_persistent_task(
        persistent_counter, persistent_task_count, persistent_macro_tiles,
        persistent_groups_n, persistent_macro_m, persistent_macro_n,
        mtile_count, ntile_count);
  }
  __syncthreads();

  int linear_tile = persistent_task_shared;
  int tile_iter = 0;
  while (linear_tile < persistent_task_count) {
    int tile_m;
    int tile_n;
    decode_persistent_task(
        linear_tile, persistent_macro_tiles, persistent_groups_n,
        persistent_macro_m, persistent_macro_n, tile_m, tile_n);
    const int ntile = ntile_count;
    const int stage_epoch_base = tile_iter * ktiles;
    const int producer_kt_begin = tile_iter == 0 ? 0 : 1;

    if (warp_id == 0 && lane0) {
      for (int kt = producer_kt_begin; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
#pragma unroll
          for (int p = 0; p < kPipes; ++p) {
            mbarrier_wait(&mma_done[p][stage], reuse_phase);
          }
        }
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
      }
      // Claim the successor as soon as W0 has issued the current mainloop.
      // Current task coordinates are already thread-local, so publishing into
      // the shared slot cannot change the tile in flight.  The existing
      // consumer-final CTA barrier below publishes this lookahead to all warps.
      persistent_task_shared = claim_next_valid_persistent_task(
          persistent_counter, persistent_task_count, persistent_macro_tiles,
          persistent_groups_n, persistent_macro_m, persistent_macro_n,
          mtile_count, ntile_count);
    }

    if (warp_id == 1 && lane0) {
      for (int kt = producer_kt_begin; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *b_smem = stage_smem + kAStageWords + kBPipeWords;
        if (stage_epoch >= kStages) {
          mbarrier_wait(
              &mma_done[1][stage],
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1));
        }
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
      }
    }

    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
#pragma unroll 1
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        const uint32_t tma_phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords + pipe * kBPipeWords;

        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);

#pragma unroll
        for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
          // B_pipe^T is the m128k16 MMA A operand.  A^T is the k16n256
          // MMA B operand.  Both descriptors address the original TMA
          // payloads; only their operand roles and instruction majors change.
          const uint64_t bt_desc = make_sw128_major_mn_smem_desc(
              smem_ptr_u32(b_smem), kk);
          const uint64_t at_desc = make_sw128_major_k_smem_desc(
              smem_ptr_u32(a_smem), kk);
          const bool input_d = (kt != 0) || (kk != 0);
          tcgen05_mma_bf16_ss(
              tmem_base + pipe * kTransposeTmemPipeStride, bt_desc, at_desc,
              idesc, input_d);
        }
        tcgen05_commit(&mma_done[pipe][stage]);
      }
      const int last_stage_epoch = stage_epoch_base + ktiles - 1;
      const int last_stage = last_stage_epoch % kStages;
      const uint32_t last_phase =
          static_cast<uint32_t>((last_stage_epoch / kStages) & 1);
      mbarrier_wait(&mma_done[pipe][last_stage], last_phase);
    }
    __syncthreads();

    // W0's lookahead claim is now visible.  Carry it into the next iteration
    // instead of claiming a second time.
    const int next_linear_tile = persistent_task_shared;

    const uint32_t acc = static_cast<uint32_t>(threadIdx.x + 0x9e3779b9u);
    if (lane0)
      warp_sinks[warp_id] = acc;
    __syncthreads();

    const int global_row_base = tile_m * kCtaM;
    const int global_col_base = tile_n * kCtaN;
    {
      const int reserved_stage = ((tile_iter + 1) * ktiles) % kStages;
      const int c_store_stage0 = (reserved_stage + 1) % kStages;
      const int c_store_stage1 = (reserved_stage + 2) % kStages;
      // Each pointer names one complete 64 KiB physical mainloop stage.
      uint32_t *c_store_smem0 = smem + c_store_stage0 * kStageWords;
      uint32_t *c_store_smem1 = smem + c_store_stage1 * kStageWords;
      issue_float_c_group_tma(
          tmem_base, &c_map, c_store_smem0, c_store_smem1, 0,
          global_row_base, global_col_base);
    }
@EARLY_PREFETCH@
    wait_float_c_group_tma();
    {
      const int reserved_stage = ((tile_iter + 1) * ktiles) % kStages;
      const int c_store_stage0 = (reserved_stage + 1) % kStages;
      const int c_store_stage1 = (reserved_stage + 2) % kStages;
      // Recompute the same explicit pointers after the overlap point so no
      // C-buffer address has a mode-dependent live range.
      uint32_t *c_store_smem0 = smem + c_store_stage0 * kStageWords;
      uint32_t *c_store_smem1 = smem + c_store_stage1 * kStageWords;
      issue_float_c_group_tma(
          tmem_base, &c_map, c_store_smem0, c_store_smem1, kCStoreBuffers,
          global_row_base, global_col_base);
    }
    wait_float_c_group_tma();
@LATE_PREFETCH@

    if (threadIdx.x == 0) {
      uint32_t tile_sink = tmem_base ^ static_cast<uint32_t>(ktiles);
#pragma unroll
      for (int w = 0; w < kWarps; ++w)
        tile_sink ^= warp_sinks[w];
      sink[tile_m * ntile + tile_n] = tile_sink;
    }
    __syncthreads();

    ++tile_iter;
    linear_tile = next_linear_tile;
  } // persistent output-tile loop
"""

SCALAR_BANNER = (
    '"nsplit_variant=transpose_mma compute=c_transpose "'
    '\n      "mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
    '\n      "phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "'
)

MODE_BANNERS = {
    "nonoverlap": (
        '"nsplit_variant=prefetch_nonoverlap compute=c_transpose "'
        '\n      "mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
        '\n      "cross_tile_prefetch=nonoverlap phase=0/0 '
        'c_store=tma_fp32_sw128 l2_promotion=none "'
    ),
    "overlap": (
        '"nsplit_variant=prefetch_overlap compute=c_transpose "'
        '\n      "mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
        '\n      "cross_tile_prefetch=overlap phase=0/0 '
        'c_store=tma_fp32_sw128 l2_promotion=none "'
    ),
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def extract_region(
    text: str, start: str, end: str, label: str, *, include_end: bool = False
) -> str:
    start_count = text.count(start)
    if start_count != 1:
        raise RuntimeError(
            f"{label}: expected one start anchor, found {start_count}"
        )
    start_pos = text.index(start)
    end_pos = text.find(end, start_pos + len(start))
    if end_pos < 0:
        raise RuntimeError(f"{label}: end anchor not found")
    if include_end:
        end_pos += len(end)
    return text[start_pos:end_pos]


def make_loop(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    early_call = PREFETCH_CALL_TEMPLATE.replace(
        "@DO_ISSUE@", "true" if mode == "overlap" else "false"
    )
    late_call = PREFETCH_CALL_TEMPLATE.replace(
        "@DO_ISSUE@", "true" if mode == "nonoverlap" else "false"
    )
    return (
        LOOP_TEMPLATE.replace("@EARLY_PREFETCH@", early_call)
        .replace("@LATE_PREFETCH@", late_call)
    )


def decode_task(
    linear_tile: int,
    persistent_macro_tiles: int,
    persistent_groups_n: int,
    persistent_macro_m: int,
    persistent_macro_n: int,
) -> tuple[int, int]:
    macro_id, local = divmod(linear_tile, persistent_macro_tiles)
    macro_m, macro_n = divmod(macro_id, persistent_groups_n)
    local_n, local_m = divmod(local, persistent_macro_m)
    return (
        macro_m * persistent_macro_m + local_m,
        macro_n * persistent_macro_n + local_n,
    )


def audit_host_mapping() -> str:
    stage_bytes = (256 * 64 // 2 + 64 * 256 // 2) * 4
    c_stage_bytes = 128 * 128 * 4
    if stage_bytes != 64 * 1024 or c_stage_bytes != stage_bytes:
        raise RuntimeError("host audit: mainloop and C stage are not 64 KiB")

    scheduler_cases = ((1, 1), (2, 2), (17, 19), (31, 33), (64, 64))
    audited_tasks = 0
    for mtile_count, ntile_count in scheduler_cases:
        macro_m = min(mtile_count, 16)
        macro_n = min(ntile_count, 16)
        groups_m = (mtile_count + macro_m - 1) // macro_m
        groups_n = (ntile_count + macro_n - 1) // macro_n
        macro_tiles = macro_m * macro_n
        task_count = groups_m * groups_n * macro_tiles
        valid: list[tuple[int, int]] = []
        for linear_tile in range(task_count):
            tile = decode_task(
                linear_tile, macro_tiles, groups_n, macro_m, macro_n
            )
            if tile[0] < mtile_count and tile[1] < ntile_count:
                valid.append(tile)
        expected = {
            (tile_m, tile_n)
            for tile_m in range(mtile_count)
            for tile_n in range(ntile_count)
        }
        if len(valid) != len(expected) or set(valid) != expected:
            raise RuntimeError(
                f"host audit: scheduler coverage failed for "
                f"{mtile_count}x{ntile_count}"
            )
        if len(set(valid)) != len(valid):
            raise RuntimeError("host audit: duplicate valid scheduler task")
        audited_tasks += len(valid)

    audited_epochs = 0
    for ktiles in (4, 8, 256):
        for tile_iter in range(6):
            stage_epoch_base = tile_iter * ktiles
            stages = [
                (stage_epoch_base + kt) % 3 for kt in range(ktiles)
            ]
            phases = [
                ((stage_epoch_base + kt) // 3) & 1
                for kt in range(ktiles)
            ]
            if len(stages) != ktiles or len(phases) != ktiles:
                raise RuntimeError("host audit: incomplete mainloop coverage")

            next_epoch = (tile_iter + 1) * ktiles
            if next_epoch != stage_epoch_base + ktiles:
                raise RuntimeError("host audit: next epoch is not contiguous")
            reserved = next_epoch % 3
            c_stages = ((reserved + 1) % 3, (reserved + 2) % 3)
            if reserved in c_stages or set(c_stages + (reserved,)) != {0, 1, 2}:
                raise RuntimeError("host audit: C buffers overlap prefetch")

            previous_epoch = next_epoch - 3
            if previous_epoch % 3 != reserved:
                raise RuntimeError("host audit: reuse barrier slot mismatch")
            previous_phase = (previous_epoch // 3) & 1
            next_phase = (next_epoch // 3) & 1
            if previous_phase == next_phase:
                raise RuntimeError("host audit: reuse phase did not toggle")

            producer_ktiles = [0] if tile_iter > 0 else []
            producer_ktiles.extend(
                range(1 if tile_iter > 0 else 0, ktiles)
            )
            if producer_ktiles != list(range(ktiles)):
                raise RuntimeError(
                    "host audit: prefetched and regular K tiles differ"
                )
            audited_epochs += ktiles

    return (
        "host_stage_audit=pass stage_bytes=65536 c_buffer_bytes=65536 "
        f"scheduler_valid_tasks={audited_tasks} audited_epochs={audited_epochs}"
    )


def audit_generated(generated: str, mode: str, expected_loop: str) -> None:
    active_call = PREFETCH_CALL_TEMPLATE.replace("@DO_ISSUE@", "true")
    inactive_call = PREFETCH_CALL_TEMPLATE.replace("@DO_ISSUE@", "false")
    required = (
        "claim_next_valid_persistent_task(",
        "decode_persistent_task(",
        "const int producer_kt_begin = tile_iter == 0 ? 0 : 1;",
        "const int reserved_stage = ((tile_iter + 1) * ktiles) % kStages;",
        "uint32_t *c_store_smem0 = smem + c_store_stage0 * kStageWords;",
        "uint32_t *c_store_smem1 = smem + c_store_stage1 * kStageWords;",
        "static_assert(kCStoreStageWords == kStageWords);",
        "const int stage_epoch = stage_epoch_base + kt;",
        "static_cast<uint32_t>((stage_epoch / kStages) & 1)",
        "const bool input_d = (kt != 0) || (kk != 0);",
        MODE_BANNERS[mode],
    )
    for fragment in required:
        if fragment not in generated:
            raise RuntimeError(f"generated audit: missing {fragment!r}")

    if generated.count(active_call) != 1:
        raise RuntimeError("generated audit: expected one active prefetch call")
    if generated.count(inactive_call) != 1:
        raise RuntimeError("generated audit: expected one inactive helper call")
    if generated.count("prefetch_next_tile_kt0(") != 3:
        raise RuntimeError(
            "generated audit: expected one helper and two dynamic calls"
        )
    if generated.count("atomicAdd(") != 1:
        raise RuntimeError("generated audit: task counter must be claimed once")
    if "store_256x256_float_tile_tma(" in generated:
        raise RuntimeError("generated audit: fixed C-store wrapper remains")
    if generated.count(
        "for (int kt = producer_kt_begin; kt < ktiles; ++kt)"
    ) != 2:
        raise RuntimeError("generated audit: producer kt0 skip is incomplete")
    if generated.count("for (int kt = 0; kt < ktiles; ++kt)") != 1:
        raise RuntimeError("generated audit: consumer must retain kt0")
    if generated.count("issue_float_c_group_tma(") != 3:
        raise RuntimeError(
            "generated audit: expected one C-group helper and two calls"
        )
    if generated.count("wait_float_c_group_tma();") != 2:
        raise RuntimeError("generated audit: both C-store groups must drain")
    if expected_loop not in generated:
        raise RuntimeError("generated audit: persistent loop differs")

    loop = extract_region(
        generated,
        LOOP_REGION_START,
        LOOP_REGION_END,
        "generated persistent loop",
        include_end=True,
    )
    active_pos = loop.index(active_call)
    inactive_pos = loop.index(inactive_call)
    first_wait = loop.index("    wait_float_c_group_tma();")
    second_wait = loop.index(
        "    wait_float_c_group_tma();",
        first_wait + len("    wait_float_c_group_tma();"),
    )
    first_commit = loop.index("    issue_float_c_group_tma(")
    early_pos = active_pos if mode == "overlap" else inactive_pos
    late_pos = inactive_pos if mode == "overlap" else active_pos
    if not first_commit < early_pos < first_wait:
        raise RuntimeError(
            "generated audit: early helper is not in group0 commit/wait"
        )
    if late_pos < second_wait:
        raise RuntimeError(
            "generated audit: late helper precedes full epilogue drain"
        )
    if mode == "overlap":
        if active_pos != early_pos:
            raise RuntimeError(
                "generated audit: overlap did not activate the early helper"
            )
    elif active_pos != late_pos:
        raise RuntimeError(
            "generated audit: nonoverlap did not activate the late helper"
        )


def generate(source: str, mode: str) -> str:
    original_store = extract_region(
        source, STORE_REGION_START, STORE_REGION_END, "C-store wrapper"
    )
    original_loop = extract_region(
        source,
        LOOP_REGION_START,
        LOOP_REGION_END,
        "persistent loop",
        include_end=True,
    )
    new_loop = make_loop(mode)

    generated = replace_once(
        source,
        STATIC_ASSERT_ANCHOR,
        PREFETCH_STATIC_ASSERTS,
        "prefetch static asserts",
    )
    generated = replace_once(
        generated,
        C_STORE_POINTER,
        C_STORE_POINTER_REPLACEMENT,
        "fixed C-store pointer",
    )
    generated = replace_once(
        generated, original_store, GROUPED_C_STORE, "C-store group split"
    )
    generated = replace_once(
        generated,
        KERNEL_ANCHOR,
        PREFETCH_HELPERS + KERNEL_ANCHOR,
        "prefetch helpers",
    )
    generated = replace_once(
        generated, original_loop, new_loop, "persistent lookahead loop"
    )
    generated = replace_once(
        generated, SCALAR_BANNER, MODE_BANNERS[mode], "mode banner"
    )

    audit_generated(generated, mode, new_loop)

    reconstructed = replace_once(
        generated, MODE_BANNERS[mode], SCALAR_BANNER, "reconstruct banner"
    )
    reconstructed = replace_once(
        reconstructed, new_loop, original_loop, "reconstruct persistent loop"
    )
    reconstructed = replace_once(
        reconstructed,
        PREFETCH_HELPERS + KERNEL_ANCHOR,
        KERNEL_ANCHOR,
        "reconstruct prefetch helpers",
    )
    reconstructed = replace_once(
        reconstructed,
        GROUPED_C_STORE,
        original_store,
        "reconstruct C-store wrapper",
    )
    reconstructed = replace_once(
        reconstructed,
        PREFETCH_STATIC_ASSERTS,
        STATIC_ASSERT_ANCHOR,
        "reconstruct static asserts",
    )
    reconstructed = replace_once(
        reconstructed,
        C_STORE_POINTER_REPLACEMENT,
        C_STORE_POINTER,
        "reconstruct fixed C-store pointer",
    )
    if reconstructed != source:
        raise RuntimeError(
            "generated audit: reconstruction differs from scalar x64 input"
        )
    return generated


def audit_mode_pair(nonoverlap: str, overlap: str) -> None:
    active_call = PREFETCH_CALL_TEMPLATE.replace("@DO_ISSUE@", "true")
    inactive_call = PREFETCH_CALL_TEMPLATE.replace("@DO_ISSUE@", "false")
    normalized_nonoverlap = nonoverlap.replace(
        MODE_BANNERS["nonoverlap"], SCALAR_BANNER, 1
    ).replace(active_call, "", 1).replace(inactive_call, "", 1)
    normalized_overlap = overlap.replace(
        MODE_BANNERS["overlap"], SCALAR_BANNER, 1
    ).replace(active_call, "", 1).replace(inactive_call, "", 1)
    if normalized_nonoverlap != normalized_overlap:
        raise RuntimeError(
            "mode-pair audit: sources differ outside banner/active call position"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "--source", dest="input", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            "refusing to patch an unaudited scalar x64 source: "
            f"expected {EXPECTED_INPUT_SHA256}, got {input_sha256}"
        )

    source = input_bytes.decode("utf-8")
    host_audit = audit_host_mapping()
    generated_by_mode = {mode: generate(source, mode) for mode in MODES}
    audit_mode_pair(
        generated_by_mode["nonoverlap"], generated_by_mode["overlap"]
    )
    generated = generated_by_mode[args.mode]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    output_sha256 = hashlib.sha256(generated.encode("utf-8")).hexdigest()
    print(host_audit)
    print(
        f"mode_pair_audit=pass "
        f"differing_only=banner,active_prefetch_call_position "
        f"variant=nsplit_prefetch_{args.mode} input_sha256={input_sha256} "
        f"output_sha256={output_sha256} output={args.output}"
    )


if __name__ == "__main__":
    main()
