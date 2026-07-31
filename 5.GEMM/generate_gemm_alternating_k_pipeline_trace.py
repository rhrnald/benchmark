#!/usr/bin/env python3
"""Generate a clock64 trace for the alternating K64/K128 GEMM.

The canonical kernel is first transformed by the audited alternating-K
generator. This pass then instruments one steady-state persistent tile. Trace
records are staged in shared memory and exported only after the epilogue.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from generate_gemm_nsplit_alternating_k import (
    EXPECTED_SOURCE_SHA256,
    generate as generate_alternating,
)


EXPECTED_ALTERNATING_SHA256 = (
    "4b7ffc3a16c60fb9a3bb36fe4e9271c84974645f1d92b8f284cb65a85920209b"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_between(
    text: str, start: str, end: str, replacement: str, label: str
) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(
            f"{label}: start={text.count(start)} end={text.count(end)}"
        )
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def instrument(alternating: str) -> str:
    text = alternating

    constants_anchor = """static constexpr int kTmemTileStride = 128;

static_assert(kCtaM == 128 || kCtaM == 256);"""
    constants_new = r"""static constexpr int kTmemTileStride = 128;

static constexpr int kTraceTargetTileIter = 8;
static constexpr int kTraceLogicalStart = 56;
static constexpr int kTraceLogicalCount = 8;

enum AlternatingTraceEvent : int {
  kTraceP0WaitPipe0 = 0,
  kTraceP0WaitPipe1,
  kTraceP0IssueA0,
  kTraceP0IssueA1,
  kTraceP0IssueB0,
  kTraceP1WaitPipe1,
  kTraceP1IssueB1,
  kTraceC2WaitB,
  kTraceC2WaitA0,
  kTraceC2WaitA1,
  kTraceC2Mma,
  kTraceC2Commit,
  kTraceC3WaitB,
  kTraceC3WaitA0,
  kTraceC3WaitA1,
  kTraceC3Mma,
  kTraceC3Commit,
  kTraceEventsPerLogicalStage,
};

static constexpr int kTraceSlotCount =
    kTraceLogicalCount * kTraceEventsPerLogicalStage;

struct alignas(16) TraceInterval {
  unsigned long long start;
  unsigned long long end;
};

struct alignas(16) TraceStageMeta {
  int logical_stage;
  int stage_epoch;
  int k64_cursor;
  int stage_k;
  int smem_slot;
};

struct alignas(16) TraceHeader {
  uint32_t magic;
  uint32_t sm_id;
  uint32_t block_idx;
  uint32_t slot_count;
  int tile_iter;
  int linear_tile;
  int tile_m;
  int tile_n;
  int logical_start;
  int logical_count;
  int events_per_stage;
  unsigned long long base_clock;
};

struct alignas(16) TraceOutput {
  TraceHeader header;
  TraceStageMeta stages[kTraceLogicalCount];
  TraceInterval slots[kTraceSlotCount];
};

static constexpr uint32_t kTraceMagic = 0x414b5452u; // "AKTR"
__device__ TraceOutput g_alternating_trace;

static_assert(kCtaM == 128 || kCtaM == 256);"""
    text = replace_once(
        text, constants_anchor, constants_new, "trace constants"
    )

    helper_anchor = (
        """__device__ __forceinline__ uint32_t smem_ptr_u32(const void *ptr) {"""
    )
    helper_new = r"""__device__ __forceinline__ unsigned long long
trace_clock64() {
#if defined(__CUDA_ARCH__)
  unsigned long long value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
#else
  return 0;
#endif
}

__device__ __forceinline__ uint32_t trace_sm_id() {
#if defined(__CUDA_ARCH__)
  uint32_t value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value) :: "memory");
  return value;
#else
  return 0;
#endif
}

__device__ __forceinline__ int trace_slot(int logical_stage, int event) {
  return (logical_stage - kTraceLogicalStart) *
             kTraceEventsPerLogicalStage +
         event;
}

__device__ __forceinline__ void trace_store(
    TraceInterval *records, bool active, int slot,
    unsigned long long start, unsigned long long end) {
  if (active)
    records[slot] = {start, end};
}

__device__ __forceinline__ uint32_t smem_ptr_u32(const void *ptr) {"""
    text = replace_once(text, helper_anchor, helper_new, "trace helpers")

    consumer_start = """template <int StageK>
__device__ __forceinline__ void consume_pipe_stage("""
    consumer_end = (
        """__device__ __forceinline__ uint32_t *alternating_stage_smem("""
    )
    consumer_new = r"""template <int StageK>
__device__ __forceinline__ void consume_pipe_stage(
    uint32_t tmem_base, uint32_t idesc, uint32_t *a_smem,
    uint32_t *b_smem, uint64_t *a_ready, uint64_t *b_ready,
    uint64_t *a_ready_long, uint64_t *mma_done, uint32_t tma_phase,
    int stage, int pipe, int k64_cursor, TraceInterval *trace_records,
    bool trace_active, int logical_stage) {
  const int event_base =
      pipe == 0 ? kTraceC2WaitB : kTraceC3WaitB;
  const unsigned long long wait_b_start =
      trace_active ? trace_clock64() : 0;
  mbarrier_wait(b_ready, tma_phase);
  const unsigned long long wait_b_end =
      trace_active ? trace_clock64() : 0;
  const unsigned long long wait_a0_start = wait_b_end;
  mbarrier_wait(a_ready, tma_phase);
  const unsigned long long wait_a0_end =
      trace_active ? trace_clock64() : 0;
  unsigned long long wait_a1_start = 0;
  unsigned long long wait_a1_end = 0;
  if (stage == 1) {
    wait_a1_start = trace_active ? trace_clock64() : 0;
    mbarrier_wait(a_ready_long, tma_phase);
    wait_a1_end = trace_active ? trace_clock64() : 0;
  }
  const unsigned long long mma_start =
      trace_active ? trace_clock64() : 0;
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
  }
  const unsigned long long mma_end =
      trace_active ? trace_clock64() : 0;
  tcgen05_commit(mma_done);
  const unsigned long long commit_end =
      trace_active ? trace_clock64() : 0;

  trace_store(trace_records, trace_active,
              trace_slot(logical_stage, event_base + 0),
              wait_b_start, wait_b_end);
  trace_store(trace_records, trace_active,
              trace_slot(logical_stage, event_base + 1),
              wait_a0_start, wait_a0_end);
  trace_store(trace_records, trace_active,
              trace_slot(logical_stage, event_base + 2),
              wait_a1_start, wait_a1_end);
  trace_store(trace_records, trace_active,
              trace_slot(logical_stage, event_base + 3),
              mma_start, mma_end);
  trace_store(trace_records, trace_active,
              trace_slot(logical_stage, event_base + 4),
              mma_end, commit_end);
}

"""
    text = replace_between(
        text,
        consumer_start,
        consumer_end,
        consumer_new,
        "instrumented consumer",
    )

    shared_anchor = """  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {"""
    shared_new = """  __shared__ uint32_t tmem_base_shared;
  __shared__ TraceHeader trace_header;
  __shared__ TraceStageMeta trace_stages[kTraceLogicalCount];
  __shared__ TraceInterval trace_records[kTraceSlotCount];

  if (threadIdx.x == 0) {"""
    text = replace_once(text, shared_anchor, shared_new, "trace shared")

    tile_anchor = """    if (tile_m >= mtile_count || tile_n >= ntile_count) {
      __syncthreads();
      continue;
    }

    int next_stage_epoch = stage_epoch_base;"""
    tile_new = """    if (tile_m >= mtile_count || tile_n >= ntile_count) {
      __syncthreads();
      continue;
    }

    const bool trace_tile =
        blockIdx.x == 0 && tile_iter == kTraceTargetTileIter;
    if (trace_tile && threadIdx.x == 0) {
      trace_header.magic = kTraceMagic;
      trace_header.sm_id = trace_sm_id();
      trace_header.block_idx = blockIdx.x;
      trace_header.slot_count = kTraceSlotCount;
      trace_header.tile_iter = tile_iter;
      trace_header.linear_tile = linear_tile;
      trace_header.tile_m = tile_m;
      trace_header.tile_n = tile_n;
      trace_header.logical_start = kTraceLogicalStart;
      trace_header.logical_count = kTraceLogicalCount;
      trace_header.events_per_stage = kTraceEventsPerLogicalStage;
      trace_header.base_clock = trace_clock64();
    }

    int next_stage_epoch = stage_epoch_base;"""
    text = replace_once(text, tile_anchor, tile_new, "trace tile header")

    producer0_start = """    if (warp_id == 0 && lane0) {"""
    producer0_end = """    if (warp_id == 1 && lane0) {"""
    producer0_new = r"""    if (warp_id == 0 && lane0) {
      int k64_cursor = 0;
      int logical_stage = 0;
      for (int stage_epoch = stage_epoch_base; k64_cursor < ktiles;
           ++stage_epoch, ++logical_stage) {
        const int stage = stage_epoch & 1;
        const int remaining = ktiles - k64_cursor;
        const bool use_k128 = stage == 1 && remaining >= 2;
        const bool trace_active =
            trace_tile && logical_stage >= kTraceLogicalStart &&
            logical_stage < kTraceLogicalStart + kTraceLogicalCount;
        uint32_t *stage_smem = alternating_stage_smem(smem, stage);
        uint32_t *a_smem = stage_smem;

        unsigned long long wait0_start =
            trace_active ? trace_clock64() : 0;
        unsigned long long wait0_end = wait0_start;
        unsigned long long wait1_end = wait0_start;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
          mbarrier_wait(&mma_done[0][stage], reuse_phase);
          wait0_end = trace_active ? trace_clock64() : 0;
          mbarrier_wait(&mma_done[1][stage], reuse_phase);
          wait1_end = trace_active ? trace_clock64() : 0;
        }

        constexpr int kK64SlabBytes =
            kCtaM * kShortStageK * sizeof(uint16_t);
        const int a_row = tile_m * kCtaM;
        const int a_col_words = k64_cursor * (kShortStageK / 2);
        mbarrier_expect_tx(&a_ready[stage], kK64SlabBytes);
        const unsigned long long a0_start =
            trace_active ? trace_clock64() : 0;
        tma_load_2d(&a64_map, smem_ptr_u32(a_smem), &a_ready[stage],
                    a_col_words, a_row);
        const unsigned long long a0_end =
            trace_active ? trace_clock64() : 0;

        unsigned long long a1_start = 0;
        unsigned long long a1_end = 0;
        if (stage == 1) {
          constexpr int kK64SlabWords = kCtaM * kShortStageK / 2;
          mbarrier_expect_tx(&a_ready_long, kK64SlabBytes);
          a1_start = trace_active ? trace_clock64() : 0;
          tma_load_2d(
              &a64_map, smem_ptr_u32(a_smem + kK64SlabWords),
              &a_ready_long,
              a_col_words + (use_k128 ? kShortStageK / 2 : 0), a_row);
          a1_end = trace_active ? trace_clock64() : 0;
        }

        uint32_t *b_smem =
            use_k128
                ? stage_smem + StageLayout<kLongStageK>::kAWords
                : stage_smem +
                      (stage == 1 ? StageLayout<kLongStageK>::kAWords
                                  : StageLayout<kShortStageK>::kAWords);
        uint64_t *b_barrier = &b_ready[0][stage];
        const int b_col_words = tile_n * (kCtaN / 2);
        const int b_k16 = k64_cursor * (kShortStageK / kMmaK);
        const int b_bytes =
            use_k128 ? StageLayout<kLongStageK>::kBPipeBytes
                     : StageLayout<kShortStageK>::kBPipeBytes;
        mbarrier_expect_tx(b_barrier, b_bytes);
        const unsigned long long b0_start =
            trace_active ? trace_clock64() : 0;
        tma_load_4d(use_k128 ? &b128_map : &b64_map,
                    smem_ptr_u32(b_smem), b_barrier, b_col_words, 0, 0,
                    b_k16);
        const unsigned long long b0_end =
            trace_active ? trace_clock64() : 0;

        trace_store(trace_records, trace_active,
                    trace_slot(logical_stage, kTraceP0WaitPipe0),
                    wait0_start, wait0_end);
        trace_store(trace_records, trace_active,
                    trace_slot(logical_stage, kTraceP0WaitPipe1),
                    wait0_end, wait1_end);
        trace_store(trace_records, trace_active,
                    trace_slot(logical_stage, kTraceP0IssueA0),
                    a0_start, a0_end);
        trace_store(trace_records, trace_active,
                    trace_slot(logical_stage, kTraceP0IssueA1),
                    a1_start, a1_end);
        trace_store(trace_records, trace_active,
                    trace_slot(logical_stage, kTraceP0IssueB0),
                    b0_start, b0_end);
        if (trace_active) {
          const int trace_index = logical_stage - kTraceLogicalStart;
          trace_stages[trace_index] = {
              logical_stage, stage_epoch, k64_cursor,
              use_k128 ? kLongStageK : kShortStageK, stage};
        }
        k64_cursor += use_k128 ? 2 : 1;
      }
    }

"""
    text = replace_between(
        text,
        producer0_start,
        producer0_end,
        producer0_new,
        "producer 0 trace",
    )

    producer1_start = """    if (warp_id == 1 && lane0) {"""
    producer1_end = """    if ((warp_id == 2 || warp_id == 3) && lane0) {"""
    producer1_new = r"""    if (warp_id == 1 && lane0) {
      int k64_cursor = 0;
      int logical_stage = 0;
      for (int stage_epoch = stage_epoch_base; k64_cursor < ktiles;
           ++stage_epoch, ++logical_stage) {
        const int stage = stage_epoch & 1;
        const int remaining = ktiles - k64_cursor;
        const bool use_k128 = stage == 1 && remaining >= 2;
        const bool trace_active =
            trace_tile && logical_stage >= kTraceLogicalStart &&
            logical_stage < kTraceLogicalStart + kTraceLogicalCount;
        uint32_t *stage_smem = alternating_stage_smem(smem, stage);
        const unsigned long long wait_start =
            trace_active ? trace_clock64() : 0;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase = static_cast<uint32_t>(
              ((stage_epoch - kStages) / kStages) & 1);
          mbarrier_wait(&mma_done[1][stage], reuse_phase);
        }
        const unsigned long long wait_end =
            trace_active ? trace_clock64() : 0;

        uint32_t *b_smem =
            use_k128
                ? stage_smem + StageLayout<kLongStageK>::kAWords +
                      StageLayout<kLongStageK>::kBPipeWords
                : stage_smem +
                      (stage == 1
                           ? StageLayout<kLongStageK>::kAWords +
                                 StageLayout<kLongStageK>::kBPipeWords
                           : StageLayout<kShortStageK>::kAWords +
                                 StageLayout<kShortStageK>::kBPipeWords);
        uint64_t *b_barrier = &b_ready[1][stage];
        const int b_col_words =
            tile_n * (kCtaN / 2) + kMmaN / 2;
        const int b_k16 = k64_cursor * (kShortStageK / kMmaK);
        const int b_bytes =
            use_k128 ? StageLayout<kLongStageK>::kBPipeBytes
                     : StageLayout<kShortStageK>::kBPipeBytes;
        mbarrier_expect_tx(b_barrier, b_bytes);
        const unsigned long long issue_start =
            trace_active ? trace_clock64() : 0;
        tma_load_4d(use_k128 ? &b128_map : &b64_map,
                    smem_ptr_u32(b_smem), b_barrier, b_col_words, 0, 0,
                    b_k16);
        const unsigned long long issue_end =
            trace_active ? trace_clock64() : 0;
        trace_store(trace_records, trace_active,
                    trace_slot(logical_stage, kTraceP1WaitPipe1),
                    wait_start, wait_end);
        trace_store(trace_records, trace_active,
                    trace_slot(logical_stage, kTraceP1IssueB1),
                    issue_start, issue_end);
        k64_cursor += use_k128 ? 2 : 1;
      }
    }

"""
    text = replace_between(
        text,
        producer1_start,
        producer1_end,
        producer1_new,
        "producer 1 trace",
    )

    text = replace_once(
        text,
        """      int k64_cursor = 0;
      int last_stage_epoch = stage_epoch_base;
      for (int stage_epoch = stage_epoch_base; k64_cursor < ktiles;
           ++stage_epoch) {""",
        """      int k64_cursor = 0;
      int logical_stage = 0;
      int last_stage_epoch = stage_epoch_base;
      for (int stage_epoch = stage_epoch_base; k64_cursor < ktiles;
           ++stage_epoch, ++logical_stage) {""",
        "consumer logical index",
    )
    text = replace_once(
        text,
        """        const bool use_k128 = stage == 1 && remaining >= 2;
        const uint32_t tma_phase =""",
        """        const bool use_k128 = stage == 1 && remaining >= 2;
        const bool trace_active =
            trace_tile && logical_stage >= kTraceLogicalStart &&
            logical_stage < kTraceLogicalStart + kTraceLogicalCount;
        const uint32_t tma_phase =""",
        "consumer trace predicate",
    )
    text = text.replace(
        """              &b_ready[pipe][stage], &a_ready_long, &mma_done[pipe][stage],
              tma_phase, stage, pipe, k64_cursor);""",
        """              &b_ready[pipe][stage], &a_ready_long, &mma_done[pipe][stage],
              tma_phase, stage, pipe, k64_cursor, trace_records,
              trace_active, logical_stage);""",
    )
    if text.count("trace_active, logical_stage);") != 2:
        raise RuntimeError("consumer calls were not both instrumented")

    epilogue_anchor = """    store_256x256_float_tile_tma(tmem_base, &c_map, c_store_smem,
                                 global_row_base, global_col_base);
    ++tile_iter;
    stage_epoch_base = next_stage_epoch;"""
    epilogue_new = """    store_256x256_float_tile_tma(tmem_base, &c_map, c_store_smem,
                                 global_row_base, global_col_base);
    if (trace_tile) {
      __syncthreads();
      for (int i = threadIdx.x; i < kTraceLogicalCount;
           i += blockDim.x)
        g_alternating_trace.stages[i] = trace_stages[i];
      for (int i = threadIdx.x; i < kTraceSlotCount;
           i += blockDim.x)
        g_alternating_trace.slots[i] = trace_records[i];
      if (threadIdx.x == 0)
        g_alternating_trace.header = trace_header;
      __syncthreads();
    }
    ++tile_iter;
    stage_epoch_base = next_stage_epoch;"""
    text = replace_once(text, epilogue_anchor, epilogue_new, "trace export")

    host_anchor = """uint16_t float_to_bf16_bits_host(float value) {"""
    host_new = r"""const char *alternating_trace_event_name(int event) {
  switch (event) {
  case kTraceP0WaitPipe0: return "p0_wait_pipe0";
  case kTraceP0WaitPipe1: return "p0_wait_pipe1";
  case kTraceP0IssueA0: return "p0_issue_a0";
  case kTraceP0IssueA1: return "p0_issue_a1";
  case kTraceP0IssueB0: return "p0_issue_b0";
  case kTraceP1WaitPipe1: return "p1_wait_pipe1";
  case kTraceP1IssueB1: return "p1_issue_b1";
  case kTraceC2WaitB: return "c2_wait_b0";
  case kTraceC2WaitA0: return "c2_wait_a0";
  case kTraceC2WaitA1: return "c2_wait_a1";
  case kTraceC2Mma: return "c2_mma";
  case kTraceC2Commit: return "c2_commit";
  case kTraceC3WaitB: return "c3_wait_b1";
  case kTraceC3WaitA0: return "c3_wait_a0";
  case kTraceC3WaitA1: return "c3_wait_a1";
  case kTraceC3Mma: return "c3_mma";
  case kTraceC3Commit: return "c3_commit";
  default: return "unknown";
  }
}

int alternating_trace_event_warp(int event) {
  if (event <= kTraceP0IssueB0)
    return 0;
  if (event <= kTraceP1IssueB1)
    return 1;
  if (event <= kTraceC2Commit)
    return 2;
  return 3;
}

void write_alternating_pipeline_trace_csv(const char *path) {
  TraceOutput trace{};
  cuda_check(cudaMemcpyFromSymbol(
      &trace, g_alternating_trace, sizeof(trace)));
  if (trace.header.magic != kTraceMagic ||
      trace.header.slot_count != kTraceSlotCount) {
    std::fprintf(stderr, "invalid alternating trace: magic=%08x slots=%u\n",
                 trace.header.magic, trace.header.slot_count);
    std::exit(EXIT_FAILURE);
  }
  FILE *csv = std::fopen(path, "w");
  if (!csv) {
    std::perror(path);
    std::exit(EXIT_FAILURE);
  }
  std::fprintf(
      csv,
      "tile_iter,linear_tile,tile_m,tile_n,logical_stage,stage_epoch,"
      "k64_cursor,stage_k,smem_slot,event,warp,start_cycle,end_cycle,"
      "duration_cycle\n");
  for (int logical_index = 0; logical_index < kTraceLogicalCount;
       ++logical_index) {
    const TraceStageMeta &meta = trace.stages[logical_index];
    for (int event = 0; event < kTraceEventsPerLogicalStage; ++event) {
      const TraceInterval &r =
          trace.slots[logical_index * kTraceEventsPerLogicalStage + event];
      if (r.start == 0 && r.end == 0)
        continue;
      std::fprintf(
          csv,
          "%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%llu,%llu,%llu\n",
          trace.header.tile_iter, trace.header.linear_tile,
          trace.header.tile_m, trace.header.tile_n, meta.logical_stage,
          meta.stage_epoch, meta.k64_cursor, meta.stage_k, meta.smem_slot,
          alternating_trace_event_name(event),
          alternating_trace_event_warp(event),
          r.start - trace.header.base_clock,
          r.end - trace.header.base_clock, r.end - r.start);
    }
  }
  std::fclose(csv);
  std::printf(
      "alternating_pipeline_trace=%s block=%u sm=%u tile_iter=%d "
      "tile=(%d,%d)\n",
      path, trace.header.block_idx, trace.header.sm_id,
      trace.header.tile_iter, trace.header.tile_m, trace.header.tile_n);
}

uint16_t float_to_bf16_bits_host(float value) {"""
    text = replace_once(text, host_anchor, host_new, "host trace writer")

    text = replace_once(
        text,
        """  std::fclose(csv);
  return 0;
}""",
        """  std::fclose(csv);
  write_alternating_pipeline_trace_csv(
      "alternating_k_pipeline_trace.csv");
  return 0;
}""",
        "trace output call",
    )
    text = replace_once(
        text,
        """"phase=0/0 consumer_wait=b_then_a c_store=tma_fp32_sw128 l2_promotion=none """,
        """"trace=clock64_alternating_k consumer_wait=b_then_a c_store=tma_fp32_sw128 l2_promotion=none """,
        "trace banner",
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
          f"refusing unaudited source: expected {EXPECTED_SOURCE_SHA256}, "
          f"got {source_sha256}"
      )
    alternating = generate_alternating(source_bytes.decode())
    alternating_sha256 = hashlib.sha256(alternating.encode()).hexdigest()
    if alternating_sha256 != EXPECTED_ALTERNATING_SHA256:
        raise SystemExit(
            "refusing unaudited alternating source: "
            f"expected {EXPECTED_ALTERNATING_SHA256}, got {alternating_sha256}"
        )
    generated = instrument(alternating)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    print(
        f"source_sha256={source_sha256} "
        f"alternating_sha256={alternating_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
