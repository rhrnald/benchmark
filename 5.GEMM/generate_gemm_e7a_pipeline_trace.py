#!/usr/bin/env python3
"""Generate a minimally instrumented per-K-stage trace of the clean E7a GEMM.

The clean performance source remains macro-free and untouched.  This generator
accepts only the audited source SHA and inserts clock64 instrumentation for an
eight-stage issue window plus three dependency-observation context stages on
block 0's ninth persistent output tile.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_BASE_SHA256 = (
    "37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def generate(base_path: Path) -> str:
    raw = base_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_BASE_SHA256:
        raise SystemExit(
            f"refusing unaudited base: sha256={digest}, "
            f"expected={EXPECTED_BASE_SHA256}"
        )
    text = raw.decode()

    constants_anchor = """static constexpr int kTmemTileStride = 128;

static_assert(kCtaM == 128 || kCtaM == 256);"""
    constants_new = """static constexpr int kTmemTileStride = 128;

// Attention-style fixed-window pipeline trace.  Only block 0's ninth valid
// persistent output tile is sampled.  Eight core K64 stages cover a full
// 3-stage-ring/parity cycle; three future context stages close the async MMA
// dependency-pass observations for that core window.
static constexpr int kPipelineTraceTargetTileIter = 8;
static constexpr int kPipelineTraceKStart = 56;
static constexpr int kPipelineTraceCoreKCount = 8;
// Keep three future stages as dependency-observation context: the software can
// first prove that MMA committed at kt is reusable when producers pass the same
// ring slot's dependency at kt + 3.
static constexpr int kPipelineTraceKCount =
    kPipelineTraceCoreKCount + kStages;

enum PipelineTraceEvent : int {
  kPipelineP0WaitM0 = 0,
  kPipelineP0WaitM1,
  kPipelineP0IssueA,
  kPipelineP0IssueB1,
  kPipelineP1WaitM0,
  kPipelineP1WaitM1,
  kPipelineP1IssueB0,
  kPipelineC2WaitA,
  kPipelineC2WaitB0,
  kPipelineC2IssueMmaB0,
  kPipelineC2WaitB1,
  kPipelineC2IssueMmaB1,
  kPipelineC2Commit,
  kPipelineC3WaitA,
  kPipelineC3WaitB0,
  kPipelineC3IssueMmaB0,
  kPipelineC3WaitB1,
  kPipelineC3IssueMmaB1,
  kPipelineC3Commit,
  kPipelineTraceEventsPerStage,
};

static constexpr int kPipelineTraceSlotCount =
    kPipelineTraceKCount * kPipelineTraceEventsPerStage;
static constexpr int kPipelineTraceValidSlotCount =
    kPipelineTraceCoreKCount * kPipelineTraceEventsPerStage +
    kStages * 4;
static_assert(kPipelineTraceEventsPerStage == 19);

struct alignas(16) PipelineTraceInterval {
  unsigned long long start = 0;
  unsigned long long end = 0;
};

struct alignas(16) PipelineTraceHeader {
  uint32_t magic = 0;
  uint32_t sm_id = 0;
  uint32_t block_idx = 0;
  uint32_t slot_count = 0;
  uint32_t valid_slot_count = 0;
  int tile_iter = -1;
  int linear_tile = -1;
  int tile_m = -1;
  int tile_n = -1;
  int ktiles = 0;
  int k_start = 0;
  int k_count = 0;
  int core_k_count = 0;
  int events_per_stage = 0;
  unsigned long long base_clock = 0;
};

struct alignas(16) PipelineTraceOutput {
  PipelineTraceHeader header;
  PipelineTraceInterval slots[kPipelineTraceSlotCount];
};

static constexpr uint32_t kPipelineTraceMagic = 0x50374b54u; // "P7KT".

static_assert(kCtaM == 128 || kCtaM == 256);"""
    text = replace_once(
        text, constants_anchor, constants_new, "trace constants"
    )

    args_anchor = """  const char *validate_pattern = "pattern";
};"""
    args_new = """  const char *validate_pattern = "pattern";
  const char *pipeline_trace_csv = nullptr;
};"""
    text = replace_once(text, args_anchor, args_new, "Args trace path")

    helper_anchor = """__device__ __forceinline__ uint32_t smem_ptr_u32(const void *ptr) {"""
    helper_new = """__device__ __forceinline__ unsigned long long
ordered_clock64() {
#if defined(__CUDA_ARCH__)
  unsigned long long value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
#else
  return 0;
#endif
}

__device__ __forceinline__ uint32_t current_sm_id() {
#if defined(__CUDA_ARCH__)
  uint32_t value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value) :: "memory");
  return value;
#else
  return 0;
#endif
}

__device__ __forceinline__ int pipeline_trace_slot(int kt, int event) {
  return (kt - kPipelineTraceKStart) * kPipelineTraceEventsPerStage + event;
}

__device__ __forceinline__ void pipeline_trace_store(
    PipelineTraceInterval *records, bool active, int slot,
    unsigned long long start, unsigned long long end) {
  if (active)
    records[slot] = {start, end};
}

__device__ __forceinline__ uint32_t smem_ptr_u32(const void *ptr) {"""
    text = replace_once(text, helper_anchor, helper_new, "device helpers")

    kernel_signature = """    const __grid_constant__ CUtensorMap c_map, uint32_t *__restrict__ sink,
    int ktiles, int mtile_count, int ntile_count) {"""
    kernel_signature_new = """    const __grid_constant__ CUtensorMap c_map, uint32_t *__restrict__ sink,
    int ktiles, int mtile_count, int ntile_count,
    PipelineTraceOutput *__restrict__ pipeline_trace) {"""
    text = replace_once(
        text, kernel_signature, kernel_signature_new, "kernel signature"
    )

    unsupported_anchor = """  (void)mtile_count;
  (void)ntile_count;
#else"""
    unsupported_new = """  (void)mtile_count;
  (void)ntile_count;
  (void)pipeline_trace;
#else"""
    text = replace_once(
        text, unsupported_anchor, unsupported_new, "unsupported arch args"
    )

    shared_anchor = """  __shared__ uint32_t warp_sinks[kWarps];
  __shared__ int persistent_task_shared;
"""
    shared_new = """  __shared__ uint32_t warp_sinks[kWarps];
  __shared__ int persistent_task_shared;
  __shared__ PipelineTraceHeader pipeline_trace_header_shared;
  __shared__ PipelineTraceInterval
      pipeline_trace_scratch[kPipelineTraceSlotCount];
"""
    text = replace_once(text, shared_anchor, shared_new, "shared trace storage")

    loop_anchor = """  int tile_iter = 0;
  while (true) {
    if (threadIdx.x == 0) {"""
    loop_new = """  int tile_iter = 0;
  while (true) {
    const bool trace_tile =
        pipeline_trace != nullptr && blockIdx.x == 0 &&
        tile_iter == kPipelineTraceTargetTileIter;
    if (threadIdx.x == 0) {"""
    text = replace_once(text, loop_anchor, loop_new, "trace tile guard")

    header_anchor = """    const int ntile = ntile_count;
    const int stage_epoch_base = tile_iter * ktiles;
"""
    header_new = """    if (trace_tile && threadIdx.x == 0) {
      pipeline_trace_header_shared.magic = kPipelineTraceMagic;
      pipeline_trace_header_shared.sm_id = current_sm_id();
      pipeline_trace_header_shared.block_idx = blockIdx.x;
      pipeline_trace_header_shared.slot_count = kPipelineTraceSlotCount;
      pipeline_trace_header_shared.valid_slot_count =
          kPipelineTraceValidSlotCount;
      pipeline_trace_header_shared.tile_iter = tile_iter;
      pipeline_trace_header_shared.linear_tile = linear_tile;
      pipeline_trace_header_shared.tile_m = tile_m;
      pipeline_trace_header_shared.tile_n = tile_n;
      pipeline_trace_header_shared.ktiles = ktiles;
      pipeline_trace_header_shared.k_start = kPipelineTraceKStart;
      pipeline_trace_header_shared.k_count = kPipelineTraceKCount;
      pipeline_trace_header_shared.core_k_count =
          kPipelineTraceCoreKCount;
      pipeline_trace_header_shared.events_per_stage =
          kPipelineTraceEventsPerStage;
      pipeline_trace_header_shared.base_clock = ordered_clock64();
    }
    const int ntile = ntile_count;
    const int stage_epoch_base = tile_iter * ktiles;
"""
    text = replace_once(text, header_anchor, header_new, "trace header")

    producer0_old = """        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
#pragma unroll
          for (int mblock = 0; mblock < kMBlocks; ++mblock) {
            mbarrier_wait(&mma_done[mblock][stage], reuse_phase);
          }
        }
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        issue_b_producer_part_tma(&b_map, b_smem, &b_ready[1][stage], tile_n,
                                  kt, 1);
      }
    }

    if (warp_id == 1 && lane0) {"""
    producer0_new = """        const bool trace_core_stage =
            trace_tile && kt >= kPipelineTraceKStart &&
            kt < kPipelineTraceKStart + kPipelineTraceCoreKCount;
        const bool trace_reuse_stage =
            trace_tile && kt >= kPipelineTraceKStart &&
            kt < kPipelineTraceKStart + kPipelineTraceKCount;
        unsigned long long wait_m0_start = 0;
        unsigned long long wait_m0_end = 0;
        unsigned long long wait_m1_end = 0;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
          wait_m0_start = trace_reuse_stage ? ordered_clock64() : 0;
          mbarrier_wait(&mma_done[0][stage], reuse_phase);
          wait_m0_end = trace_reuse_stage ? ordered_clock64() : 0;
          mbarrier_wait(&mma_done[1][stage], reuse_phase);
          wait_m1_end = trace_reuse_stage ? ordered_clock64() : 0;
        }
        const unsigned long long a_start =
            trace_core_stage ? wait_m1_end : 0;
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        const unsigned long long a_end =
            trace_core_stage ? ordered_clock64() : 0;
        const unsigned long long b1_start = a_end;
        issue_b_producer_part_tma(&b_map, b_smem, &b_ready[1][stage], tile_n,
                                  kt, 1);
        const unsigned long long b1_end =
            trace_core_stage ? ordered_clock64() : 0;

        pipeline_trace_store(
            pipeline_trace_scratch, trace_reuse_stage,
            pipeline_trace_slot(kt, kPipelineP0WaitM0),
            wait_m0_start, wait_m0_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_reuse_stage,
            pipeline_trace_slot(kt, kPipelineP0WaitM1),
            wait_m0_end, wait_m1_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, kPipelineP0IssueA),
            a_start, a_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, kPipelineP0IssueB1),
            b1_start, b1_end);
      }
    }

    if (warp_id == 1 && lane0) {"""
    text = replace_once(
        text, producer0_old, producer0_new, "producer warp 0 instrumentation"
    )

    producer1_old = """        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
#pragma unroll
          for (int mblock = 0; mblock < kMBlocks; ++mblock) {
            mbarrier_wait(&mma_done[mblock][stage], reuse_phase);
          }
        }
        issue_b_producer_part_tma(&b_map, b_smem, &b_ready[0][stage], tile_n,
                                  kt, 0);
      }
    }

    if ((warp_id == 2 || warp_id == 3) && lane0) {"""
    producer1_new = """        const bool trace_core_stage =
            trace_tile && kt >= kPipelineTraceKStart &&
            kt < kPipelineTraceKStart + kPipelineTraceCoreKCount;
        const bool trace_reuse_stage =
            trace_tile && kt >= kPipelineTraceKStart &&
            kt < kPipelineTraceKStart + kPipelineTraceKCount;
        unsigned long long wait_m0_start = 0;
        unsigned long long wait_m0_end = 0;
        unsigned long long wait_m1_end = 0;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
          wait_m0_start = trace_reuse_stage ? ordered_clock64() : 0;
          mbarrier_wait(&mma_done[0][stage], reuse_phase);
          wait_m0_end = trace_reuse_stage ? ordered_clock64() : 0;
          mbarrier_wait(&mma_done[1][stage], reuse_phase);
          wait_m1_end = trace_reuse_stage ? ordered_clock64() : 0;
        }
        const unsigned long long b0_start =
            trace_core_stage ? wait_m1_end : 0;
        issue_b_producer_part_tma(&b_map, b_smem, &b_ready[0][stage], tile_n,
                                  kt, 0);
        const unsigned long long b0_end =
            trace_core_stage ? ordered_clock64() : 0;

        pipeline_trace_store(
            pipeline_trace_scratch, trace_reuse_stage,
            pipeline_trace_slot(kt, kPipelineP1WaitM0),
            wait_m0_start, wait_m0_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_reuse_stage,
            pipeline_trace_slot(kt, kPipelineP1WaitM1),
            wait_m0_end, wait_m1_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, kPipelineP1IssueB0),
            b0_start, b0_end);
      }
    }

    if ((warp_id == 2 || warp_id == 3) && lane0) {"""
    text = replace_once(
        text, producer1_old, producer1_new, "producer warp 1 instrumentation"
    )

    consumer_setup_old = """        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords;

        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[0][stage], tma_phase);
#pragma unroll"""
    consumer_setup_new = """        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords;
        const bool trace_core_stage =
            trace_tile && kt >= kPipelineTraceKStart &&
            kt < kPipelineTraceKStart + kPipelineTraceCoreKCount;
        const int consumer_event_base =
            kPipelineC2WaitA + mblock * 6;

        const unsigned long long wait_a_start =
            trace_core_stage ? ordered_clock64() : 0;
        mbarrier_wait(&a_ready[stage], tma_phase);
        const unsigned long long wait_a_end =
            trace_core_stage ? ordered_clock64() : 0;
        mbarrier_wait(&b_ready[0][stage], tma_phase);
        const unsigned long long wait_b0_end =
            trace_core_stage ? ordered_clock64() : 0;
#pragma unroll"""
    text = replace_once(
        text, consumer_setup_old, consumer_setup_new, "consumer waits A/B0"
    )

    consumer_mid_old = """          tcgen05_mma_bf16_ss(c_taddr, a_desc, b0_desc, idesc, input_d);
        }
        mbarrier_wait(&b_ready[1][stage], tma_phase);
#pragma unroll"""
    consumer_mid_new = """          tcgen05_mma_bf16_ss(c_taddr, a_desc, b0_desc, idesc, input_d);
        }
        const unsigned long long mma_b0_end =
            trace_core_stage ? ordered_clock64() : 0;
        mbarrier_wait(&b_ready[1][stage], tma_phase);
        const unsigned long long wait_b1_end =
            trace_core_stage ? ordered_clock64() : 0;
#pragma unroll"""
    text = replace_once(
        text, consumer_mid_old, consumer_mid_new, "consumer MMA B0/B1 wait"
    )

    consumer_commit_old = """          tcgen05_mma_bf16_ss(c_taddr, a_desc, b0_desc, idesc, input_d);
        }
        tcgen05_commit(&mma_done[mblock][stage]);
      }
      const int last_stage_epoch"""
    consumer_commit_new = """          tcgen05_mma_bf16_ss(c_taddr, a_desc, b0_desc, idesc, input_d);
        }
        const unsigned long long mma_b1_end =
            trace_core_stage ? ordered_clock64() : 0;
        tcgen05_commit(&mma_done[mblock][stage]);
        const unsigned long long commit_end =
            trace_core_stage ? ordered_clock64() : 0;

        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, consumer_event_base + 0),
            wait_a_start, wait_a_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, consumer_event_base + 1),
            wait_a_end, wait_b0_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, consumer_event_base + 2),
            wait_b0_end, mma_b0_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, consumer_event_base + 3),
            mma_b0_end, wait_b1_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, consumer_event_base + 4),
            wait_b1_end, mma_b1_end);
        pipeline_trace_store(
            pipeline_trace_scratch, trace_core_stage,
            pipeline_trace_slot(kt, consumer_event_base + 5),
            mma_b1_end, commit_end);
      }
      const int last_stage_epoch"""
    text = replace_once(
        text, consumer_commit_old, consumer_commit_new, "consumer commit"
    )

    export_anchor = """    __syncthreads();

    ++tile_iter;
  } // persistent output-tile loop"""
    export_new = """    __syncthreads();

    if (trace_tile) {
      // All pipeline timestamps are finished before the shared-to-global
      // export, so global trace traffic cannot sit between selected events.
      __syncthreads();
      for (int slot = threadIdx.x; slot < kPipelineTraceSlotCount;
           slot += blockDim.x) {
        pipeline_trace->slots[slot] = pipeline_trace_scratch[slot];
      }
      if (threadIdx.x == 0)
        pipeline_trace->header = pipeline_trace_header_shared;
      __syncthreads();
    }

    ++tile_iter;
  } // persistent output-tile loop"""
    text = replace_once(text, export_anchor, export_new, "trace export")

    usage_old = """              "[--validate] [--validate-size N] "
              "[--validate-pattern pattern|ones]\\n","""
    usage_new = """              "[--pipeline-trace-csv PATH] "
              "[--validate] [--validate-size N] "
              "[--validate-pattern pattern|ones]\\n","""
    text = replace_once(text, usage_old, usage_new, "usage")

    parse_old = """    } else if (std::strcmp(argv[i], "--input-init") == 0) {"""
    parse_new = """    } else if (std::strcmp(argv[i], "--pipeline-trace-csv") == 0) {
      args.pipeline_trace_csv = need_arg("--pipeline-trace-csv");
    } else if (std::strcmp(argv[i], "--input-init") == 0) {"""
    text = replace_once(text, parse_old, parse_new, "trace CLI")

    launch_old = """void launch_gemm_kernel(dim3 grid, const CUtensorMap &a_map,
                        const CUtensorMap &b_map, const CUtensorMap &c_map,
                        uint32_t *d_sink, int ktiles, int mtile, int ntile) {
  gemm256_bf16_16k_kernel<<<grid, kThreads, kDynamicSmemBytes>>>(
      a_map, b_map, c_map, d_sink, ktiles, mtile, ntile);
}"""
    launch_new = """void launch_gemm_kernel(
    dim3 grid, const CUtensorMap &a_map, const CUtensorMap &b_map,
    const CUtensorMap &c_map, uint32_t *d_sink, int ktiles, int mtile,
    int ntile, PipelineTraceOutput *pipeline_trace = nullptr) {
  gemm256_bf16_16k_kernel<<<grid, kThreads, kDynamicSmemBytes>>>(
      a_map, b_map, c_map, d_sink, ktiles, mtile, ntile, pipeline_trace);
}"""
    text = replace_once(text, launch_old, launch_new, "kernel launcher")

    host_anchor = """uint16_t float_to_bf16_bits_host(float value) {"""
    host_new = r"""const char *pipeline_trace_event_name(int event) {
  switch (event) {
  case kPipelineP0WaitM0: return "p0_wait_mma_m0";
  case kPipelineP0WaitM1: return "p0_wait_mma_m1";
  case kPipelineP0IssueA: return "p0_prepare_and_issue_tma_a";
  case kPipelineP0IssueB1: return "p0_prepare_and_issue_tma_b1";
  case kPipelineP1WaitM0: return "p1_wait_mma_m0";
  case kPipelineP1WaitM1: return "p1_wait_mma_m1";
  case kPipelineP1IssueB0: return "p1_prepare_and_issue_tma_b0";
  case kPipelineC2WaitA: return "c2_wait_tma_a";
  case kPipelineC2WaitB0: return "c2_wait_tma_b0";
  case kPipelineC2IssueMmaB0: return "c2_prepare_and_issue_mma_b0";
  case kPipelineC2WaitB1: return "c2_wait_tma_b1";
  case kPipelineC2IssueMmaB1: return "c2_prepare_and_issue_mma_b1";
  case kPipelineC2Commit: return "c2_commit_mma";
  case kPipelineC3WaitA: return "c3_wait_tma_a";
  case kPipelineC3WaitB0: return "c3_wait_tma_b0";
  case kPipelineC3IssueMmaB0: return "c3_prepare_and_issue_mma_b0";
  case kPipelineC3WaitB1: return "c3_wait_tma_b1";
  case kPipelineC3IssueMmaB1: return "c3_prepare_and_issue_mma_b1";
  case kPipelineC3Commit: return "c3_commit_mma";
  default: return "unknown";
  }
}

int pipeline_trace_event_warp(int event) {
  if (event <= kPipelineP0IssueB1)
    return 0;
  if (event <= kPipelineP1IssueB0)
    return 1;
  if (event <= kPipelineC2Commit)
    return 2;
  return 3;
}

bool pipeline_trace_is_reuse_wait(int event) {
  return event == kPipelineP0WaitM0 ||
         event == kPipelineP0WaitM1 ||
         event == kPipelineP1WaitM0 ||
         event == kPipelineP1WaitM1;
}

bool pipeline_trace_record_expected(int kt, int event) {
  const bool core =
      kt < kPipelineTraceKStart + kPipelineTraceCoreKCount;
  return core || pipeline_trace_is_reuse_wait(event);
}

void run_pipeline_trace_case(const char *trace_csv, int input_init_mode) {
  constexpr int size = kBenchmarkSize;
  const int m = size;
  const int n = size;
  const int k = size;
  const int mtile = m / kCtaM;
  const int ntile = n / kCtaN;
  const int ktiles = k / kStageK;
  const int ctas = mtile * ntile;
  const size_t a_words = static_cast<size_t>(m) * k / 2;
  const size_t b_words = static_cast<size_t>(k) * n / 2;

  uint32_t *d_a = nullptr;
  uint32_t *d_b = nullptr;
  uint32_t *d_sink = nullptr;
  float *d_c = nullptr;
  PipelineTraceOutput *d_trace = nullptr;
  cuda_check(cudaMalloc(&d_a, a_words * sizeof(uint32_t)));
  cuda_check(cudaMalloc(&d_b, b_words * sizeof(uint32_t)));
  cuda_check(
      cudaMalloc(&d_sink, (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  cuda_check(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
  cuda_check(cudaMalloc(&d_trace, sizeof(PipelineTraceOutput)));
  cuda_check(cudaMemset(d_c, 0, static_cast<size_t>(m) * n * sizeof(float)));
  initialize_bf16_inputs(d_a, a_words, d_b, b_words, input_init_mode);
  cuda_check(cudaMemset(d_sink, 0,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  cuda_check(cudaMemset(d_trace, 0, sizeof(PipelineTraceOutput)));
  cuda_check(cudaDeviceSynchronize());

  CUtensorMap a_map{}, b_map{}, c_map{};
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n);
  encode_c_row_major_float_tma_map(&c_map, d_c, m, n);
  set_gemm_kernel_attribute();

  const dim3 grid(std::min(kPersistentCtas, ctas), 1, 1);
  auto launch = [&](PipelineTraceOutput *trace) {
    cuda_check(cudaMemsetAsync(d_sink + ctas, 0, sizeof(uint32_t)));
    launch_gemm_kernel(grid, a_map, b_map, c_map, d_sink, ktiles, mtile,
                       ntile, trace);
    cuda_check(cudaGetLastError());
  };

  // Establish steady/cache state with one untraced full launch, then let one
  // traced launch write exactly one CTA/window.  The selected-event semantics
  // follow the attention trace, while its launch warmup policy is independent.
  launch(nullptr);
  cuda_check(cudaDeviceSynchronize());
  cuda_check(cudaMemset(d_trace, 0, sizeof(PipelineTraceOutput)));
  launch(d_trace);
  cuda_check(cudaDeviceSynchronize());

  PipelineTraceOutput result{};
  cuda_check(cudaMemcpy(&result, d_trace, sizeof(result),
                        cudaMemcpyDeviceToHost));
  if (result.header.magic != kPipelineTraceMagic ||
      result.header.slot_count != kPipelineTraceSlotCount ||
      result.header.valid_slot_count != kPipelineTraceValidSlotCount ||
      result.header.k_start != kPipelineTraceKStart ||
      result.header.k_count != kPipelineTraceKCount ||
      result.header.core_k_count != kPipelineTraceCoreKCount) {
    std::fprintf(stderr,
                 "pipeline trace was not recorded: magic=%08x slots=%u/%u "
                 "k_start=%d k_count=%d core_k_count=%d\n",
                 result.header.magic, result.header.slot_count,
                 result.header.valid_slot_count,
                 result.header.k_start, result.header.k_count,
                 result.header.core_k_count);
    std::exit(EXIT_FAILURE);
  }

  unsigned long long first_clock = ~0ull;
  for (int slot = 0; slot < kPipelineTraceSlotCount; ++slot) {
    const int iter_index = slot / kPipelineTraceEventsPerStage;
    const int event = slot - iter_index * kPipelineTraceEventsPerStage;
    const int kt = kPipelineTraceKStart + iter_index;
    if (!pipeline_trace_record_expected(kt, event))
      continue;
    const PipelineTraceInterval &record = result.slots[slot];
    if (record.end <= record.start) {
      std::fprintf(stderr,
                   "invalid pipeline trace slot=%d start=%llu end=%llu\n",
                   slot, record.start, record.end);
      std::exit(EXIT_FAILURE);
    }
    first_clock = std::min(first_clock, record.start);
  }

  FILE *csv = std::fopen(trace_csv, "w");
  if (!csv) {
    std::perror(trace_csv);
    std::exit(EXIT_FAILURE);
  }
  std::fprintf(
      csv,
      "size,input,sm_id,block_idx,tile_iter,linear_tile,tile_m,tile_n,"
      "ktiles,k_start,k_count,core_k_count,base_clock,kt,stage_epoch,"
      "ring_stage,phase,event_id,related_kt,"
      "event,warp,start_raw,end_raw,start_rel,end_rel,cycles\n");
  for (int slot = 0; slot < kPipelineTraceSlotCount; ++slot) {
    const int iter_index = slot / kPipelineTraceEventsPerStage;
    const int event = slot - iter_index * kPipelineTraceEventsPerStage;
    const int kt = kPipelineTraceKStart + iter_index;
    if (!pipeline_trace_record_expected(kt, event))
      continue;
    const int stage_epoch = result.header.tile_iter * ktiles + kt;
    const int ring_stage = stage_epoch % kStages;
    const bool reuse_wait = pipeline_trace_is_reuse_wait(event);
    const int phase = reuse_wait
                          ? ((stage_epoch - kStages) / kStages) & 1
                          : (stage_epoch / kStages) & 1;
    const int related_kt = reuse_wait ? kt - kStages : -1;
    const PipelineTraceInterval &record = result.slots[slot];
    std::fprintf(
        csv,
        "%d,%s,%u,%u,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%d,%d,%d,%d,%d,"
        "%d,%s,%d,"
        "%llu,%llu,%llu,%llu,%llu\n",
        size, input_init_mode_name(input_init_mode), result.header.sm_id,
        result.header.block_idx, result.header.tile_iter,
        result.header.linear_tile, result.header.tile_m,
        result.header.tile_n, result.header.ktiles, result.header.k_start,
        result.header.k_count, result.header.core_k_count,
        first_clock, kt, stage_epoch, ring_stage, phase, event,
        related_kt,
        pipeline_trace_event_name(event), pipeline_trace_event_warp(event),
        record.start, record.end,
        record.start - first_clock,
        record.end - first_clock, record.end - record.start);
  }
  std::fclose(csv);

  std::printf(
      "pipeline_trace_csv=%s size=%d input=%s block=%u sm=%u "
      "tile_iter=%d linear_tile=%d tile=(%d,%d) k=[%d,%d] "
      "records=%u slots=%u\n",
      trace_csv, size, input_init_mode_name(input_init_mode),
      result.header.block_idx, result.header.sm_id, result.header.tile_iter,
      result.header.linear_tile, result.header.tile_m, result.header.tile_n,
      result.header.k_start,
      result.header.k_start + result.header.core_k_count - 1,
      result.header.valid_slot_count,
      result.header.slot_count);

  cuda_check(cudaFree(d_a));
  cuda_check(cudaFree(d_b));
  cuda_check(cudaFree(d_sink));
  cuda_check(cudaFree(d_c));
  cuda_check(cudaFree(d_trace));
}

uint16_t float_to_bf16_bits_host(float value) {"""
    text = replace_once(text, host_anchor, host_new, "host trace path")

    main_anchor = """  if (args.validate) {
    const ValidateResult r ="""
    main_new = """  if (args.pipeline_trace_csv != nullptr) {
    run_pipeline_trace_case(args.pipeline_trace_csv, args.input_init_mode);
    return 0;
  }

  if (args.validate) {
    const ValidateResult r ="""
    text = replace_once(text, main_anchor, main_new, "main trace dispatch")

    banner_anchor = """// Clean 16K dense working default."""
    banner_new = """// Attention-style per-K-stage pipeline trace generated from the exact clean
// E7a 16K source.  Do not use this instrumented binary for TFLOP/s.
//
// Clean 16K dense working default."""
    text = replace_once(text, banner_anchor, banner_new, "source banner")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("5.GEMM/baseline/gemm256_bf16_16k.cu"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    generated = generate(args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    print(
        f"generated={args.output} base={args.base} "
        f"base_sha256={EXPECTED_BASE_SHA256}"
    )


if __name__ == "__main__":
    main()
