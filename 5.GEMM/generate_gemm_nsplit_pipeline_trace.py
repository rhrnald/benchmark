#!/usr/bin/env python3
"""Generate a clock64 pipeline trace from the canonical N-split GEMM.

Only block 0's ninth persistent output tile is traced.  Timestamps are staged
in shared memory and exported after the tile, so global trace traffic does not
sit between selected pipeline events.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "eb90e11322c3af9adba08ed6262fe585bd0b37c9ba2fb1750c802e393c6b6dc2"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def generate(source: str, wait_order: str) -> str:
    text = source
    constants_anchor = """static constexpr int kTmemTileStride = 128;

static_assert(kCtaM == 128 || kCtaM == 256);"""
    constants_new = r"""static constexpr int kTmemTileStride = 128;

static constexpr int kTraceTargetTileIter = 8;
static constexpr int kTraceKStart = 56;
static constexpr int kTraceCoreKCount = 8;
static constexpr int kTraceKCount = kTraceCoreKCount + kStages;

enum TraceEvent : int {
  kTraceP0WaitPipe0 = 0,
  kTraceP0WaitPipe1,
  kTraceP0IssueA,
  kTraceP0IssueB0,
  kTraceP1WaitPipe1,
  kTraceP1IssueB1,
  kTraceC2WaitA,
  kTraceC2WaitB,
  kTraceC2Mma,
  kTraceC2Commit,
  kTraceC3WaitA,
  kTraceC3WaitB,
  kTraceC3Mma,
  kTraceC3Commit,
  kTraceEventsPerStage,
};

static constexpr int kTraceStageSlots = kTraceKCount * kTraceEventsPerStage;
static constexpr int kTraceEpilogueSlot = kTraceStageSlots;
static constexpr int kTraceSlotCount = kTraceStageSlots + 1;

struct alignas(16) TraceInterval {
  unsigned long long start;
  unsigned long long end;
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
  int k_start;
  int k_count;
  int core_k_count;
  int events_per_stage;
  unsigned long long base_clock;
};

struct alignas(16) TraceOutput {
  TraceHeader header;
  TraceInterval slots[kTraceSlotCount];
};

static constexpr uint32_t kTraceMagic = 0x4e535054u; // "NSPT"
__device__ TraceOutput g_pipeline_trace;

static_assert(kCtaM == 128 || kCtaM == 256);"""
    text = replace_once(text, constants_anchor, constants_new, "trace constants")

    helper_anchor = """__device__ __forceinline__ uint32_t smem_ptr_u32(const void *ptr) {"""
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

__device__ __forceinline__ int trace_slot(int kt, int event) {
  return (kt - kTraceKStart) * kTraceEventsPerStage + event;
}

__device__ __forceinline__ void trace_store(
    TraceInterval *records, bool active, int slot,
    unsigned long long start, unsigned long long end) {
  if (active)
    records[slot] = {start, end};
}

__device__ __forceinline__ uint32_t smem_ptr_u32(const void *ptr) {"""
    text = replace_once(text, helper_anchor, helper_new, "trace helpers")

    shared_anchor = """  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {"""
    shared_new = """  __shared__ uint32_t tmem_base_shared;
  __shared__ TraceHeader trace_header;
  __shared__ TraceInterval trace_records[kTraceSlotCount];

  if (threadIdx.x == 0) {"""
    text = replace_once(text, shared_anchor, shared_new, "trace shared storage")

    tile_anchor = """    const int ntile = ntile_count;
    const int stage_epoch_base = tile_iter * ktiles;
"""
    tile_new = """    const bool trace_tile =
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
      trace_header.k_start = kTraceKStart;
      trace_header.k_count = kTraceKCount;
      trace_header.core_k_count = kTraceCoreKCount;
      trace_header.events_per_stage = kTraceEventsPerStage;
      trace_header.base_clock = trace_clock64();
      trace_records[kTraceEpilogueSlot] = {0, 0};
    }
    const int ntile = ntile_count;
    const int stage_epoch_base = tile_iter * ktiles;
"""
    text = replace_once(text, tile_anchor, tile_new, "trace tile header")

    producer0_anchor = """        if (stage_epoch >= kStages) {
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
"""
    producer0_new = """        const bool trace_core =
            trace_tile && kt >= kTraceKStart &&
            kt < kTraceKStart + kTraceCoreKCount;
        const bool trace_context =
            trace_tile && kt >= kTraceKStart &&
            kt < kTraceKStart + kTraceKCount;
        unsigned long long wait0_start = 0;
        unsigned long long wait0_end = 0;
        unsigned long long wait1_end = 0;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
          wait0_start = trace_context ? trace_clock64() : 0;
          mbarrier_wait(&mma_done[0][stage], reuse_phase);
          wait0_end = trace_context ? trace_clock64() : 0;
          mbarrier_wait(&mma_done[1][stage], reuse_phase);
          wait1_end = trace_context ? trace_clock64() : 0;
        }
        const unsigned long long a_start = trace_core ? trace_clock64() : 0;
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        const unsigned long long a_end = trace_core ? trace_clock64() : 0;
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
        const unsigned long long b0_end = trace_core ? trace_clock64() : 0;
        trace_store(trace_records, trace_context,
                    trace_slot(kt, kTraceP0WaitPipe0),
                    wait0_start, wait0_end);
        trace_store(trace_records, trace_context,
                    trace_slot(kt, kTraceP0WaitPipe1),
                    wait0_end, wait1_end);
        trace_store(trace_records, trace_core,
                    trace_slot(kt, kTraceP0IssueA), a_start, a_end);
        trace_store(trace_records, trace_core,
                    trace_slot(kt, kTraceP0IssueB0), a_end, b0_end);
"""
    text = replace_once(
        text, producer0_anchor, producer0_new, "producer 0 trace"
    )

    producer1_anchor = """        if (stage_epoch >= kStages) {
          mbarrier_wait(
              &mma_done[1][stage],
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1));
        }
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
"""
    producer1_new = """        const bool trace_core =
            trace_tile && kt >= kTraceKStart &&
            kt < kTraceKStart + kTraceCoreKCount;
        const bool trace_context =
            trace_tile && kt >= kTraceKStart &&
            kt < kTraceKStart + kTraceKCount;
        const unsigned long long wait_start =
            trace_context ? trace_clock64() : 0;
        if (stage_epoch >= kStages) {
          mbarrier_wait(
              &mma_done[1][stage],
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1));
        }
        const unsigned long long wait_end =
            trace_context ? trace_clock64() : 0;
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
        const unsigned long long issue_end =
            trace_core ? trace_clock64() : 0;
        trace_store(trace_records, trace_context,
                    trace_slot(kt, kTraceP1WaitPipe1),
                    wait_start, wait_end);
        trace_store(trace_records, trace_core,
                    trace_slot(kt, kTraceP1IssueB1),
                    wait_end, issue_end);
"""
    text = replace_once(
        text, producer1_anchor, producer1_new, "producer 1 trace"
    )

    consumer_wait_anchor = """        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);

#pragma unroll
"""
    if wait_order == "a-first":
        consumer_wait_new = """        const bool trace_core =
            trace_tile && kt >= kTraceKStart &&
            kt < kTraceKStart + kTraceCoreKCount;
        const int event_base =
            pipe == 0 ? kTraceC2WaitA : kTraceC3WaitA;
        const unsigned long long wait_a_start =
            trace_core ? trace_clock64() : 0;
        mbarrier_wait(&a_ready[stage], tma_phase);
        const unsigned long long wait_a_end =
            trace_core ? trace_clock64() : 0;
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);
        const unsigned long long wait_b_end =
            trace_core ? trace_clock64() : 0;

#pragma unroll
"""
        mma_start_name = "wait_b_end"
    else:
        consumer_wait_new = """        const bool trace_core =
            trace_tile && kt >= kTraceKStart &&
            kt < kTraceKStart + kTraceCoreKCount;
        const int event_base =
            pipe == 0 ? kTraceC2WaitA : kTraceC3WaitA;
        const unsigned long long wait_b_start =
            trace_core ? trace_clock64() : 0;
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);
        const unsigned long long wait_b_end =
            trace_core ? trace_clock64() : 0;
        const unsigned long long wait_a_start = wait_b_end;
        mbarrier_wait(&a_ready[stage], tma_phase);
        const unsigned long long wait_a_end =
            trace_core ? trace_clock64() : 0;

#pragma unroll
"""
        mma_start_name = "wait_a_end"
    text = replace_once(
        text, consumer_wait_anchor, consumer_wait_new, "consumer wait trace"
    )

    consumer_commit_anchor = """        tcgen05_commit(&mma_done[pipe][stage]);
      }
"""
    if wait_order == "a-first":
        wait_store = """        trace_store(trace_records, trace_core,
                    trace_slot(kt, event_base + 0),
                    wait_a_start, wait_a_end);
        trace_store(trace_records, trace_core,
                    trace_slot(kt, event_base + 1),
                    wait_a_end, wait_b_end);"""
    else:
        wait_store = """        trace_store(trace_records, trace_core,
                    trace_slot(kt, event_base + 0),
                    wait_a_start, wait_a_end);
        trace_store(trace_records, trace_core,
                    trace_slot(kt, event_base + 1),
                    wait_b_start, wait_b_end);"""
    consumer_commit_new = """        const unsigned long long mma_end =
            trace_core ? trace_clock64() : 0;
        tcgen05_commit(&mma_done[pipe][stage]);
        const unsigned long long commit_end =
            trace_core ? trace_clock64() : 0;
__WAIT_STORE__
        trace_store(trace_records, trace_core,
                    trace_slot(kt, event_base + 2),
                    __MMA_START__, mma_end);
        trace_store(trace_records, trace_core,
                    trace_slot(kt, event_base + 3),
                    mma_end, commit_end);
      }
""".replace("__WAIT_STORE__", wait_store).replace("__MMA_START__", mma_start_name)
    text = replace_once(
        text, consumer_commit_anchor, consumer_commit_new, "consumer commit trace"
    )

    epilogue_anchor = """    __syncthreads();

    const int global_row_base = tile_m * kCtaM;
    const int global_col_base = tile_n * kCtaN;
    store_256x256_float_tile_tma(tmem_base, &c_map, c_store_smem,
                                 global_row_base,
                                 global_col_base);
    ++tile_iter;
"""
    epilogue_new = """    __syncthreads();

    const unsigned long long epilogue_start =
        trace_tile && threadIdx.x == 0 ? trace_clock64() : 0;
    const int global_row_base = tile_m * kCtaM;
    const int global_col_base = tile_n * kCtaN;
    store_256x256_float_tile_tma(tmem_base, &c_map, c_store_smem,
                                 global_row_base,
                                 global_col_base);
    if (trace_tile && threadIdx.x == 0)
      trace_records[kTraceEpilogueSlot] = {epilogue_start, trace_clock64()};
    if (trace_tile) {
      __syncthreads();
      for (int slot = threadIdx.x; slot < kTraceSlotCount;
           slot += blockDim.x)
        g_pipeline_trace.slots[slot] = trace_records[slot];
      if (threadIdx.x == 0)
        g_pipeline_trace.header = trace_header;
      __syncthreads();
    }
    ++tile_iter;
"""
    text = replace_once(text, epilogue_anchor, epilogue_new, "trace export")

    host_anchor = """uint16_t float_to_bf16_bits_host(float value) {"""
    host_new = r"""const char *trace_event_name(int event) {
  switch (event) {
  case kTraceP0WaitPipe0: return "p0_wait_pipe0";
  case kTraceP0WaitPipe1: return "p0_wait_pipe1";
  case kTraceP0IssueA: return "p0_issue_a";
  case kTraceP0IssueB0: return "p0_issue_b0";
  case kTraceP1WaitPipe1: return "p1_wait_pipe1";
  case kTraceP1IssueB1: return "p1_issue_b1";
  case kTraceC2WaitA: return "c2_wait_a";
  case kTraceC2WaitB: return "c2_wait_b0";
  case kTraceC2Mma: return "c2_mma";
  case kTraceC2Commit: return "c2_commit";
  case kTraceC3WaitA: return "c3_wait_a";
  case kTraceC3WaitB: return "c3_wait_b1";
  case kTraceC3Mma: return "c3_mma";
  case kTraceC3Commit: return "c3_commit";
  default: return "unknown";
  }
}

int trace_event_warp(int event) {
  if (event <= kTraceP0IssueB0)
    return 0;
  if (event <= kTraceP1IssueB1)
    return 1;
  if (event <= kTraceC2Commit)
    return 2;
  return 3;
}

void write_pipeline_trace_csv(const char *path) {
  TraceOutput trace{};
  cuda_check(cudaMemcpyFromSymbol(&trace, g_pipeline_trace, sizeof(trace)));
  if (trace.header.magic != kTraceMagic ||
      trace.header.slot_count != kTraceSlotCount) {
    std::fprintf(stderr, "invalid trace header: magic=%08x slots=%u\n",
                 trace.header.magic, trace.header.slot_count);
    std::exit(EXIT_FAILURE);
  }
  FILE *csv = std::fopen(path, "w");
  if (!csv) {
    std::perror(path);
    std::exit(EXIT_FAILURE);
  }
  std::fprintf(csv,
      "tile_iter,linear_tile,tile_m,tile_n,kt,event,warp,"
      "start_cycle,end_cycle,duration_cycle\n");
  for (int slot = 0; slot < kTraceStageSlots; ++slot) {
    const TraceInterval &r = trace.slots[slot];
    if (r.start == 0 && r.end == 0)
      continue;
    const int kt_index = slot / kTraceEventsPerStage;
    const int event = slot - kt_index * kTraceEventsPerStage;
    const int kt = kTraceKStart + kt_index;
    std::fprintf(csv, "%d,%d,%d,%d,%d,%s,%d,%llu,%llu,%llu\n",
        trace.header.tile_iter, trace.header.linear_tile,
        trace.header.tile_m, trace.header.tile_n, kt,
        trace_event_name(event), trace_event_warp(event),
        r.start - trace.header.base_clock,
        r.end - trace.header.base_clock, r.end - r.start);
  }
  const TraceInterval &epilogue = trace.slots[kTraceEpilogueSlot];
  std::fprintf(csv, "%d,%d,%d,%d,-1,epilogue_total,0,%llu,%llu,%llu\n",
      trace.header.tile_iter, trace.header.linear_tile,
      trace.header.tile_m, trace.header.tile_n,
      epilogue.start - trace.header.base_clock,
      epilogue.end - trace.header.base_clock,
      epilogue.end - epilogue.start);
  std::fclose(csv);
  std::printf("pipeline_trace=%s block=%u sm=%u tile_iter=%d tile=(%d,%d)\n",
              path, trace.header.block_idx, trace.header.sm_id,
              trace.header.tile_iter, trace.header.tile_m, trace.header.tile_n);
}

uint16_t float_to_bf16_bits_host(float value) {"""
    text = replace_once(text, host_anchor, host_new, "host trace writer")

    main_anchor = """  std::fclose(csv);
  return 0;
}"""
    main_new = """  std::fclose(csv);
  write_pipeline_trace_csv("nsplit_pipeline_trace.csv");
  return 0;
}"""
    text = replace_once(text, main_anchor, main_new, "trace main output")

    banner_anchor = (
        '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "'
    )
    text = replace_once(
        text,
        banner_anchor,
        f'"trace=clock64_nsplit wait_order={wait_order} '
        'c_store=tma_fp32_sw128 l2_promotion=none "',
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
    parser.add_argument(
        "--wait-order", choices=("a-first", "b-first"), default="a-first"
    )
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "refusing unaudited source: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )
    generated = generate(source_bytes.decode(), args.wait_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    print(
        f"source_sha256={source_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
