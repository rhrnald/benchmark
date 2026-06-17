// Core CUDA kernels for the fused attention benchmark.
// Included by main.cu so templated kernels remain in the same translation unit.

#ifndef ATTENTION_EPILOGUE_CHUNK_COLS
#define ATTENTION_EPILOGUE_CHUNK_COLS 16
#endif

#ifndef ATTENTION_EPILOGUE_O_IN_S_SMEM
#define ATTENTION_EPILOGUE_O_IN_S_SMEM 1
#endif

#ifndef ATTENTION_PERSISTENT
#define ATTENTION_PERSISTENT 0
#endif

#ifndef ATTENTION_PERSISTENT_NOINLINE_ROLES
#define ATTENTION_PERSISTENT_NOINLINE_ROLES 0
#endif
#if ATTENTION_PERSISTENT && ATTENTION_PERSISTENT_NOINLINE_ROLES
#define ATTENTION_PIPE_ROLE_INLINE __noinline__
#else
#define ATTENTION_PIPE_ROLE_INLINE __forceinline__
#endif

#ifndef ATTENTION_PERSISTENT_DESC_GEN
#define ATTENTION_PERSISTENT_DESC_GEN ATTENTION_PERSISTENT
#endif

#ifndef ATTENTION_PERSISTENT_OVERLAP
#define ATTENTION_PERSISTENT_OVERLAP 0
#endif
#if ATTENTION_PERSISTENT_OVERLAP && !ATTENTION_PERSISTENT
#error "ATTENTION_PERSISTENT_OVERLAP requires ATTENTION_PERSISTENT"
#endif
#ifndef ATTENTION_PERSISTENT_OVERLAP_PREFETCH
#define ATTENTION_PERSISTENT_OVERLAP_PREFETCH ATTENTION_PERSISTENT_OVERLAP
#endif

#ifndef ATTENTION_PERSISTENT_OVERLAP_EARLY
#define ATTENTION_PERSISTENT_OVERLAP_EARLY 0
#endif
#if ATTENTION_PERSISTENT_OVERLAP_EARLY && !ATTENTION_PERSISTENT_OVERLAP_PREFETCH
#error "ATTENTION_PERSISTENT_OVERLAP_EARLY requires ATTENTION_PERSISTENT_OVERLAP_PREFETCH"
#endif

#ifndef ATTENTION_PERSISTENT_OVERLAP_O_IN_V
#define ATTENTION_PERSISTENT_OVERLAP_O_IN_V 0
#endif
#if ATTENTION_PERSISTENT_OVERLAP_O_IN_V && !ATTENTION_PERSISTENT
#error "ATTENTION_PERSISTENT_OVERLAP_O_IN_V requires ATTENTION_PERSISTENT"
#endif

#ifndef ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
#define ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE 0
#endif
#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE && \
    !(ATTENTION_PERSISTENT_OVERLAP_O_IN_V && ATTENTION_PERSISTENT_OVERLAP_EARLY)
#error "ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE requires O_IN_V and EARLY"
#endif

#ifndef ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
#define ATTENTION_PERSISTENT_OVERLAP_QK_PEEL 0
#endif
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL && !ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
#error "ATTENTION_PERSISTENT_OVERLAP_QK_PEEL requires ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE"
#endif

#ifndef ATTENTION_PEEL_PREARM
#define ATTENTION_PEEL_PREARM 0
#endif

#ifndef ATTENTION_PEEL_HB
#define ATTENTION_PEEL_HB 0
#endif
#if ATTENTION_PEEL_HB
__device__ unsigned int g_peel_hb[16];
#define PEEL_HB_SET(slot, val)                              \
  do {                                                      \
    if (blockIdx.x == 0 && lane0) {                         \
      g_peel_hb[(slot)] = (val);                            \
      __threadfence();                                      \
    }                                                       \
  } while (0)
#else
#define PEEL_HB_SET(slot, val) do { } while (0)
#endif

#ifndef ATTENTION_PEEL_ISO
#define ATTENTION_PEEL_ISO 0
#endif
#if ATTENTION_PEEL_ISO && !ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
#error "ATTENTION_PEEL_ISO requires ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE"
#endif
#ifndef ATTENTION_ISO_ARRAY
#define ATTENTION_ISO_ARRAY 0
#endif

#ifndef ATTENTION_PEEL_ROLE_NOSKIP
#define ATTENTION_PEEL_ROLE_NOSKIP 0
#endif

#ifndef ATTENTION_PEEL_PIPE0_ONLY
#define ATTENTION_PEEL_PIPE0_ONLY 0
#endif

#ifndef ATTENTION_PEEL_DURING_DRAIN
#define ATTENTION_PEEL_DURING_DRAIN 0
#endif

#ifndef ATTENTION_PEEL_AFTER_SYNC
#define ATTENTION_PEEL_AFTER_SYNC 0
#endif

#ifndef ATTENTION_PEEL_DD_PRESYNC
#define ATTENTION_PEEL_DD_PRESYNC 0
#endif

#ifndef ATTENTION_PEEL_DD_SINGLE_ISSUER
#define ATTENTION_PEEL_DD_SINGLE_ISSUER 0
#endif

#ifndef ATTENTION_PEEL_DD_SERIAL_COMMIT
#define ATTENTION_PEEL_DD_SERIAL_COMMIT 0
#endif

#ifndef ATTENTION_ROW_MAX_ONLY
#define ATTENTION_ROW_MAX_ONLY 0
#endif

#ifndef ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
#define ATTENTION_FIRST_ITER_ROW_MAX_SHIFT 1
#endif

#ifndef ATTENTION_FIRST_ITER_APPLY_SHIFT
#define ATTENTION_FIRST_ITER_APPLY_SHIFT ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
#endif

#ifndef ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE
#define ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE ATTENTION_FIRST_ITER_APPLY_SHIFT
#endif

#ifndef ATTENTION_FIRST_ITER_COMPUTE_MAX
#define ATTENTION_FIRST_ITER_COMPUTE_MAX ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
#endif

#ifndef ATTENTION_ROW_SUM_RARE_UPDATE
#define ATTENTION_ROW_SUM_RARE_UPDATE 1
#endif

#ifndef ATTENTION_ROW_SUM_UPDATE_LIMIT
#define ATTENTION_ROW_SUM_UPDATE_LIMIT 256.0f
#endif

#ifndef ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS
#define ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS 1
#endif

#ifndef ATTENTION_CROSS_PIPE_PHASE
#define ATTENTION_CROSS_PIPE_PHASE 0
#endif

#ifndef ATTENTION_PIPE1_TMA_HEAD_DELAY_CYCLES
#define ATTENTION_PIPE1_TMA_HEAD_DELAY_CYCLES 1728
#endif

#ifndef ATTENTION_PIPE1_TMA_HEAD_DELAY_MIN_REPEATS
#define ATTENTION_PIPE1_TMA_HEAD_DELAY_MIN_REPEATS 0
#endif

#ifndef ATTENTION_PIPE1_TMA_HEAD_MARKER
#define ATTENTION_PIPE1_TMA_HEAD_MARKER 0
#endif

#ifndef ATTENTION_PIPE1_QK_HEAD_DELAY_CYCLES
#define ATTENTION_PIPE1_QK_HEAD_DELAY_CYCLES 0
#endif

#ifndef ATTENTION_SPLIT_V_TMA
#define ATTENTION_SPLIT_V_TMA 1
#endif

#ifndef ATTENTION_SPLIT_V_H0_WITH_K_TMA
#define ATTENTION_SPLIT_V_H0_WITH_K_TMA 1
#endif

#ifndef ATTENTION_SPLIT_V_H0_BEFORE_K_TMA
#define ATTENTION_SPLIT_V_H0_BEFORE_K_TMA 0
#endif

#ifndef ATTENTION_SKIP_V_H0_READY_WAIT
#define ATTENTION_SKIP_V_H0_READY_WAIT 1
#endif

#ifndef ATTENTION_SKIP_V_H1_READY_WAIT
#define ATTENTION_SKIP_V_H1_READY_WAIT 1
#endif

#ifndef ATTENTION_SKIP_V_H0_READY_WAIT_STEADY
#define ATTENTION_SKIP_V_H0_READY_WAIT_STEADY ATTENTION_SKIP_V_H0_READY_WAIT
#endif

#ifndef ATTENTION_SKIP_V_H0_READY_WAIT_TAIL
#define ATTENTION_SKIP_V_H0_READY_WAIT_TAIL ATTENTION_SKIP_V_H0_READY_WAIT
#endif

#ifndef ATTENTION_SKIP_V_H1_READY_WAIT_STEADY
#define ATTENTION_SKIP_V_H1_READY_WAIT_STEADY ATTENTION_SKIP_V_H1_READY_WAIT
#endif

#ifndef ATTENTION_SKIP_V_H1_READY_WAIT_TAIL
#define ATTENTION_SKIP_V_H1_READY_WAIT_TAIL ATTENTION_SKIP_V_H1_READY_WAIT
#endif

#ifndef ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER
#define ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER 9
#endif

#if ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER != 0 && \
    (ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER <= 8 || \
     ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER >= 12)
#error "ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER must be 0 or 9..11"
#endif

#ifndef ATTENTION_MINIMAL_TMA_GAP_TRACE
#define ATTENTION_MINIMAL_TMA_GAP_TRACE 0
#endif

#define ATTENTION_CROSS_PHASE_TMA_K_SERIAL 1
#define ATTENTION_CROSS_PHASE_TMA_V_SERIAL 2
#define ATTENTION_CROSS_PHASE_TMA_KV_SERIAL 3
#define ATTENTION_CROSS_PHASE_QK_AFTER_PIPE0 4
#define ATTENTION_CROSS_PHASE_TMA_K_ISSUE 5
#define ATTENTION_CROSS_PHASE_TMA_V_ISSUE 6
#define ATTENTION_CROSS_PHASE_TMA_KV_ISSUE 7
#define ATTENTION_CROSS_PHASE_QK_ISSUE 8

__device__ __forceinline__ void attention_clock_delay(unsigned long long cycles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
#else
  (void)cycles;
#endif
}

__device__ __forceinline__ void cross_pipe_signal(volatile uint32_t* counter,
                                                  uint32_t value) {
  __threadfence_block();
  *counter = value;
}

__device__ __forceinline__ void cross_pipe_wait_at_least(
    volatile const uint32_t* counter,
    uint32_t value) {
  while (*counter < value) {
  }
}

__global__ void fill_packed_bf16(uint32_t* ptr, size_t words, uint32_t seed) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < words) {
    ptr[i] = 0x3f803f80u ^ ((static_cast<uint32_t>(i) + seed * 977u) & 0x000f000fu);
  }
}

static constexpr int kRealAttentionD = 128;
static constexpr int kRealAttentionThreads = kRealAttentionD;

struct RealAttentionParams {
  const uint16_t* __restrict__ q;
  const uint16_t* __restrict__ k;
  const uint16_t* __restrict__ v;
  uint16_t* __restrict__ o;
  int B;
  int Hq;
  int Hkv;
  int Sq;
  int Skv;
  int D;
  int causal;
  float softmax_scale;
};

__host__ __device__ __forceinline__ bool real_attention_key_is_valid(
    int q_idx, int k_idx, int sq, int skv, int causal) {
  if (!causal) return true;
  // Bottom-right aligned causal masking.  When Sq == Skv this is k_idx <= q_idx.
  // When Skv > Sq, the query tile is treated as the suffix of the KV sequence.
  const int causal_limit = q_idx + (skv - sq);
  return k_idx <= causal_limit;
}

__host__ __device__ __forceinline__ int real_attention_hkv_for_hq(
    int hq, int hq_count, int hkv_count) {
  if (hkv_count <= 1) return 0;
  if (hkv_count == hq_count) return hq;
  const int group = hq_count / hkv_count;
  const int mapped = group > 0 ? hq / group : 0;
  return mapped < hkv_count ? mapped : hkv_count - 1;
}

__device__ __forceinline__ float real_bf16_to_float_device(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t real_float_to_bf16_device(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float real_block_dot_d128(float q_lane,
                                                     const uint16_t* k_row,
                                                     int tid,
                                                     float* scratch) {
  scratch[tid] = q_lane * real_bf16_to_float_device(k_row[tid]);
  __syncthreads();
#pragma unroll
  for (int stride = kRealAttentionD / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  return scratch[0];
}

__global__ __launch_bounds__(kRealAttentionThreads, 1)
void real_attention_bf16_d128_kernel(RealAttentionParams p) {
  if (p.D != kRealAttentionD) return;

  const int tid = threadIdx.x;
  const int q_idx = static_cast<int>(blockIdx.x);
  const int bhq = static_cast<int>(blockIdx.y);
  if (tid >= kRealAttentionD || q_idx >= p.Sq || bhq >= p.B * p.Hq) return;

  const int b = bhq / p.Hq;
  const int hq = bhq - b * p.Hq;
  const int hkv = real_attention_hkv_for_hq(hq, p.Hq, p.Hkv);

  const size_t q_base = ((static_cast<size_t>(b) * p.Hq + hq) * p.Sq + q_idx) * p.D;
  const size_t kv_base = (static_cast<size_t>(b) * p.Hkv + hkv) * p.Skv * p.D;
  const size_t o_base = ((static_cast<size_t>(b) * p.Hq + hq) * p.Sq + q_idx) * p.D;

  const float q_lane = real_bf16_to_float_device(p.q[q_base + tid]);

  __shared__ float scratch[kRealAttentionD];
  __shared__ float row_max_s;
  __shared__ float denom_s;
  __shared__ float weight_s;

  float row_max = -3.4028234663852886e+38f;
  for (int k_idx = 0; k_idx < p.Skv; ++k_idx) {
    if (!real_attention_key_is_valid(q_idx, k_idx, p.Sq, p.Skv, p.causal)) continue;
    const uint16_t* k_row = p.k + kv_base + static_cast<size_t>(k_idx) * p.D;
    const float dot = real_block_dot_d128(q_lane, k_row, tid, scratch);
    if (tid == 0) {
      const float score = dot * p.softmax_scale;
      row_max = fmaxf(row_max, score);
    }
  }
  if (tid == 0) {
    row_max_s = row_max;
    denom_s = 0.0f;
  }
  __syncthreads();

  if (row_max_s == -3.4028234663852886e+38f) {
    p.o[o_base + tid] = real_float_to_bf16_device(0.0f);
    return;
  }

  float out_acc = 0.0f;
  for (int k_idx = 0; k_idx < p.Skv; ++k_idx) {
    if (!real_attention_key_is_valid(q_idx, k_idx, p.Sq, p.Skv, p.causal)) continue;
    const uint16_t* k_row = p.k + kv_base + static_cast<size_t>(k_idx) * p.D;
    const float dot = real_block_dot_d128(q_lane, k_row, tid, scratch);
    if (tid == 0) {
      const float score = dot * p.softmax_scale;
      weight_s = expf(score - row_max_s);
      denom_s += weight_s;
    }
    __syncthreads();
    const uint16_t* v_row = p.v + kv_base + static_cast<size_t>(k_idx) * p.D;
    out_acc += weight_s * real_bf16_to_float_device(v_row[tid]);
    __syncthreads();
  }

  const float denom = denom_s;
  const float out = denom > 0.0f ? out_acc / denom : 0.0f;
  p.o[o_base + tid] = real_float_to_bf16_device(out);
}

template <int kFixedKTiles>
__device__ ATTENTION_PIPE_ROLE_INLINE void attention_pv_pipe_role(
    const CUtensorMap* k_map,
    const CUtensorMap* v_map,
    uint32_t* const (&k_smem)[kPipeCount],
    uint32_t* const (&v_smem)[kPipeCount],
    uint64_t (&k_ready)[kPipeCount],
    uint64_t (&v_ready)[kPipeCount],
    uint64_t (&v_h1_ready)[kPipeCount],
    uint64_t (&qk_done)[kPipeCount],
    uint64_t (&pv_done)[kPipeCount],
    uint32_t (&k_issue_gen)[kPipeCount],
    uint32_t (&v_issue_gen)[kPipeCount],
    uint64_t* tma_head_marker,
    unsigned long long* k_tma_start_shared,
    int pipe,
    int loop_repeats,
    int loop_k_tiles,
    int kv_tile_base,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    bool k_prefetched,
#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
    bool wait_prev_store,
#endif
#if ATTENTION_PEEL_HB
    bool hb_peeled,
#endif
    int lane) {
  const int role_warp_id = 2 + pipe;
  const bool lane0 = lane == 0;
#if !ATTENTION_CLOCK_TRACE
  (void)role_warp_id;
  (void)k_tma_start_shared;
#endif
  int iter = pipe;
  int local = 0;
#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
  if (pipe == 0 && lane0 && wait_prev_store) {
    tma_store_wait_group_read();
  }
  __syncwarp();
#endif
  if (iter < loop_repeats) {
#if ATTENTION_PIPE1_TMA_HEAD_MARKER
    if (pipe == 1) {
      mbarrier_wait(tma_head_marker, 0);
    }
#endif
#if ATTENTION_PIPE1_TMA_HEAD_DELAY_CYCLES > 0
    if constexpr (kFixedKTiles == 0 ||
                  kFixedKTiles >= ATTENTION_PIPE1_TMA_HEAD_DELAY_MIN_REPEATS) {
      if (pipe == 1) {
        attention_clock_delay(ATTENTION_PIPE1_TMA_HEAD_DELAY_CYCLES);
      }
    }
#endif
    const int k_tile = local_k_tile_for_iter<kFixedKTiles>(iter, loop_k_tiles);
    const int global_k_tile = kv_tile_base + k_tile;
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_K_SERIAL || \
    ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_SERIAL
    if (pipe == 1) {
      mbarrier_wait(&k_ready[0], 0);
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_K_ISSUE || \
    ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
    if (pipe == 1) {
      cross_pipe_wait_at_least(&k_issue_gen[0], 1);
    }
#endif
#if ATTENTION_CLOCK_TRACE
    const int trace_idx = iter - clock_trace_start;
    const bool trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
        trace_idx < clock_trace_iters;
    const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
    unsigned long long k_tma_start = 0ull;
    if (trace_iter) {
      k_tma_start = clock64();
      k_tma_start_shared[pipe * 2] = k_tma_start;
      __threadfence_block();
    }
#endif
    if (!k_prefetched) {
      issue_k_tma_tile(k_map, k_smem[pipe], &k_ready[pipe], global_k_tile,
                       lane0);
    }
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_K_ISSUE || \
    ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
    if (pipe == 0 && lane0) {
      cross_pipe_signal(&k_issue_gen[0], 1);
    }
#endif
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      const unsigned long long k_tma_issue_end = clock64();
      write_clock_trace_record(clock_trace,
                               trace_slot_base + kClockTraceKTmaIssueSlot,
                               kClockTraceKTmaIssue, iter, pipe, role_warp_id,
                               -1, -1, k_tma_start, k_tma_issue_end,
                               clock_trace_base);
    }
#endif
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      begin_clock_trace_record(clock_trace, trace_slot_base + 1,
                               kClockTraceKTma, iter, pipe, role_warp_id, -1,
                               -1, k_tma_start, clock_trace_base);
    }
#endif
#if ATTENTION_CLOCK_TRACE
    unsigned long long v_tma_start = 0ull;
    if (trace_iter && lane0) {
      v_tma_start = clock64();
#if ATTENTION_SPLIT_V_TMA
      begin_clock_trace_record(clock_trace, trace_slot_base + 3,
                               kClockTraceVTma, iter, pipe, role_warp_id, -1,
                               0, v_tma_start, clock_trace_base);
#else
      begin_clock_trace_record(clock_trace, trace_slot_base + 3,
                               kClockTraceVTma, iter, pipe, role_warp_id, -1,
                               -1, v_tma_start, clock_trace_base);
#endif
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_V_SERIAL || \
    ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_SERIAL
    if (pipe == 1) {
      mbarrier_wait(&v_ready[0], 0);
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_V_ISSUE || \
    ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
    if (pipe == 1) {
      cross_pipe_wait_at_least(&v_issue_gen[0], 1);
    }
#endif
#if ATTENTION_SPLIT_V_TMA
    issue_v_tma_half_tile(v_map, v_smem[pipe], &v_ready[pipe], global_k_tile,
                          0, lane0);
#if ATTENTION_CLOCK_TRACE
    if (trace_iter && lane0) {
      write_clock_trace_record(clock_trace,
                               trace_slot_base + kClockTraceVTmaIssueSlot,
                               kClockTraceVTmaIssue, iter, pipe, role_warp_id,
                               -1, 0, v_tma_start, clock64(),
                               clock_trace_base);
      const unsigned long long v_tma_h1_start = clock64();
      begin_clock_trace_record(clock_trace, trace_slot_base + 48,
                               kClockTraceVTma, iter, pipe, role_warp_id, -1,
                               1, v_tma_h1_start, clock_trace_base);
      v_tma_start = v_tma_h1_start;
    }
#endif
    issue_v_tma_half_tile(v_map, v_smem[pipe], &v_h1_ready[pipe], global_k_tile,
                          1, lane0);
#else
    issue_v_tma_tile(v_map, v_smem[pipe], &v_ready[pipe], global_k_tile,
                     lane0);
#endif
#if ATTENTION_PIPE1_TMA_HEAD_MARKER
    if (pipe == 0 && lane0) {
      mbarrier_arrive(tma_head_marker);
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_V_ISSUE || \
    ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
    if (pipe == 0 && lane0) {
      cross_pipe_signal(&v_issue_gen[0], 1);
    }
#endif
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      write_clock_trace_record(clock_trace,
#if ATTENTION_SPLIT_V_TMA
                               trace_slot_base + 49,
                               kClockTraceVTmaIssue, iter, pipe, role_warp_id,
                               -1, 1, v_tma_start, clock64(),
                               clock_trace_base);
#else
                               trace_slot_base + kClockTraceVTmaIssueSlot,
                               kClockTraceVTmaIssue, iter, pipe, role_warp_id,
                               -1, -1, v_tma_start, clock64(),
                               clock_trace_base);
#endif
    }
#endif
  }
  for (; iter < loop_repeats; iter += kActivePipeStride, ++local) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    const int next_iter = iter + kActivePipeStride;
    const bool has_next_iter = next_iter < loop_repeats;
    const uint32_t next_phase = static_cast<uint32_t>((local + 1) & 1);
#if !ATTENTION_CLOCK_TRACE
    (void)next_phase;
#endif
#if ATTENTION_CLOCK_TRACE
    int next_trace_slot_base = 0;
    bool next_trace_iter = false;
    unsigned long long k_tma_start = 0ull;
    unsigned long long k_tma_issue_end_for_gap = 0ull;
#endif
    const int global_v_tile =
        kv_tile_base + local_k_tile_for_iter<kFixedKTiles>(iter, loop_k_tiles);
    if (has_next_iter) {
#if ATTENTION_CLOCK_TRACE
      unsigned long long qk_wait_start = 0ull;
      if (clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
          iter >= clock_trace_start &&
          iter < clock_trace_start + clock_trace_iters) {
        qk_wait_start = clock64();
      }
#endif
#if ATTENTION_PEEL_HB
      if (hb_peeled && local == 0) PEEL_HB_SET(6 + pipe, 1u);
#endif
      mbarrier_wait(&qk_done[pipe], phase);
#if ATTENTION_PEEL_HB
      if (hb_peeled && local == 0) PEEL_HB_SET(6 + pipe, 2u);
#endif
#if ATTENTION_CLOCK_TRACE
      if (qk_wait_start != 0ull) {
        write_clock_trace_record(clock_trace,
                                 (iter - clock_trace_start) *
                                         kClockTraceSlotsPerIter +
                                     kClockTraceSyncBase,
                                 kClockTraceSync, iter, pipe, role_warp_id, -1,
                                 0, qk_wait_start, clock64(),
                                 clock_trace_base);
      }
#endif
      const int k_tile =
          local_k_tile_for_iter<kFixedKTiles>(next_iter, loop_k_tiles);
      const int global_k_tile = kv_tile_base + k_tile;
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_K_SERIAL || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_SERIAL
      if (pipe == 1) {
        mbarrier_wait(&k_ready[0], next_phase);
      }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_K_ISSUE || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
      if (pipe == 1) {
        cross_pipe_wait_at_least(&k_issue_gen[0],
                                 static_cast<uint32_t>(local + 2));
      }
#endif
#if ATTENTION_CLOCK_TRACE
      const int next_trace_idx = next_iter - clock_trace_start;
      next_trace_iter =
          clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
          next_trace_idx >= 0 && next_trace_idx < clock_trace_iters;
      next_trace_slot_base = next_trace_idx * kClockTraceSlotsPerIter;
      if (next_trace_iter) {
#if !(ATTENTION_SPLIT_V_TMA && ATTENTION_SPLIT_V_H0_WITH_K_TMA && ATTENTION_SPLIT_V_H0_BEFORE_K_TMA)
        k_tma_start = clock64();
        k_tma_start_shared[pipe * 2 + next_phase] = k_tma_start;
        __threadfence_block();
#endif
      }
#endif
#if ATTENTION_SPLIT_V_TMA && ATTENTION_SPLIT_V_H0_WITH_K_TMA && ATTENTION_SPLIT_V_H0_BEFORE_K_TMA
      if (local > 0) {
#if ATTENTION_CLOCK_TRACE
        unsigned long long v_tma_h0_start = 0ull;
#if ATTENTION_MINIMAL_TMA_GAP_TRACE
        v_tma_h0_start = clock64();
#endif
        const int trace_idx = iter - clock_trace_start;
        const bool trace_iter =
            clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
            trace_idx >= 0 && trace_idx < clock_trace_iters;
        const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
        if (trace_iter) {
#if !ATTENTION_MINIMAL_TMA_GAP_TRACE
          v_tma_h0_start = clock64();
#endif
          begin_clock_trace_record(clock_trace, trace_slot_base + 3,
                                   kClockTraceVTma, iter, pipe, role_warp_id,
                                   -1, 0, v_tma_h0_start, clock_trace_base);
        }
#endif
        issue_v_tma_half_tile(v_map, v_smem[pipe], &v_ready[pipe],
                              global_v_tile, 0, lane0);
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) {
          write_clock_trace_record(clock_trace,
                                   trace_slot_base + kClockTraceVTmaIssueSlot,
                                   kClockTraceVTmaIssue, iter, pipe,
                                   role_warp_id, -1, 0, v_tma_h0_start,
                                   clock64(), clock_trace_base);
        }
#endif
      }
#if ATTENTION_CLOCK_TRACE
      if (next_trace_iter) {
        k_tma_start = clock64();
        k_tma_start_shared[pipe * 2 + next_phase] = k_tma_start;
        __threadfence_block();
      }
#endif
#endif
      issue_k_tma_tile(k_map, k_smem[pipe], &k_ready[pipe], global_k_tile,
                       lane0);
#if ATTENTION_CLOCK_TRACE && ATTENTION_MINIMAL_TMA_GAP_TRACE
      if (next_trace_iter) {
        k_tma_issue_end_for_gap = clock64();
      }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_K_ISSUE || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
      if (pipe == 0 && lane0) {
        cross_pipe_signal(&k_issue_gen[0],
                          static_cast<uint32_t>(local + 2));
      }
#endif
#if ATTENTION_CLOCK_TRACE && \
    !(ATTENTION_MINIMAL_TMA_GAP_TRACE && ATTENTION_SPLIT_V_TMA && \
      ATTENTION_SPLIT_V_H0_WITH_K_TMA && !ATTENTION_SPLIT_V_H0_BEFORE_K_TMA)
      if (next_trace_iter) {
        const unsigned long long k_tma_issue_end = clock64();
        write_clock_trace_record(clock_trace,
                                 next_trace_slot_base +
                                     kClockTraceKTmaIssueSlot,
                                 kClockTraceKTmaIssue, next_iter, pipe,
                                 role_warp_id, -1, -1, k_tma_start,
                                 k_tma_issue_end,
                                 clock_trace_base);
      }
#endif
#if ATTENTION_CLOCK_TRACE && \
    !(ATTENTION_MINIMAL_TMA_GAP_TRACE && ATTENTION_SPLIT_V_TMA && \
      ATTENTION_SPLIT_V_H0_WITH_K_TMA && !ATTENTION_SPLIT_V_H0_BEFORE_K_TMA)
      if (next_trace_iter) {
        begin_clock_trace_record(clock_trace, next_trace_slot_base + 1,
                                 kClockTraceKTma, next_iter, pipe,
                                 role_warp_id, -1, -1, k_tma_start,
                                 clock_trace_base);
      }
#endif
#if ATTENTION_SPLIT_V_TMA && ATTENTION_SPLIT_V_H0_WITH_K_TMA && !ATTENTION_SPLIT_V_H0_BEFORE_K_TMA
      if (local > 0) {
#if ATTENTION_CLOCK_TRACE
        const int trace_idx = iter - clock_trace_start;
        const bool trace_iter =
            clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
            trace_idx >= 0 && trace_idx < clock_trace_iters;
        const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
        unsigned long long v_tma_h0_start = 0ull;
        if (trace_iter) {
          v_tma_h0_start = clock64();
          begin_clock_trace_record(clock_trace, trace_slot_base + 3,
                                   kClockTraceVTma, iter, pipe, role_warp_id,
                                   -1, 0, v_tma_h0_start, clock_trace_base);
        }
#endif
        issue_v_tma_half_tile(v_map, v_smem[pipe], &v_ready[pipe],
                              global_v_tile, 0, lane0);
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) {
#if ATTENTION_MINIMAL_TMA_GAP_TRACE
          if (next_trace_iter) {
            write_clock_trace_record(clock_trace, trace_slot_base + 50,
                                     kClockTraceSync, iter, pipe,
                                     role_warp_id, -1, 3,
                                     k_tma_issue_end_for_gap,
                                     v_tma_h0_start, clock_trace_base);
          }
#endif
          write_clock_trace_record(clock_trace,
                                   trace_slot_base + kClockTraceVTmaIssueSlot,
                                   kClockTraceVTmaIssue, iter, pipe,
                                   role_warp_id, -1, 0, v_tma_h0_start,
                                   clock64(), clock_trace_base);
        }
#endif
      }
#endif
    }
#if ATTENTION_CLOCK_TRACE
    const int trace_idx = iter - clock_trace_start;
    const bool trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
        trace_idx < clock_trace_iters;
    const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
    unsigned long long v_tma_start = 0ull;
#endif
    if (local > 0) {
#if ATTENTION_CLOCK_TRACE
      const int done_iter = iter - kActivePipeStride;
      const int done_trace_idx = done_iter - clock_trace_start;
      const bool done_trace_iter =
          clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
          done_trace_idx >= 0 && done_trace_idx < clock_trace_iters;
      const unsigned long long pv_done_wait_start =
          done_trace_iter ? clock64() : 0ull;
#endif
      mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
#if ATTENTION_CLOCK_TRACE
      if (done_trace_iter) {
        write_clock_trace_record(clock_trace,
                                 done_trace_idx * kClockTraceSlotsPerIter +
                                     kClockTraceSyncBase + 1,
                                 kClockTraceSync, done_iter, pipe,
                                 role_warp_id, -1, 1, pv_done_wait_start,
                                 clock64(), clock_trace_base);
        end_clock_trace_record(clock_trace,
                               done_trace_idx * kClockTraceSlotsPerIter + 4,
                               clock64(), clock_trace_base);
      }
#endif
    }
#if ATTENTION_CLOCK_TRACE
    if (trace_iter && lane0 && local > 0) {
#if ATTENTION_SPLIT_V_TMA && ATTENTION_SPLIT_V_H0_WITH_K_TMA
      if (has_next_iter) {
        v_tma_start = clock64();
        begin_clock_trace_record(clock_trace, trace_slot_base + 48,
                                 kClockTraceVTma, iter, pipe, role_warp_id, -1,
                                 1, v_tma_start, clock_trace_base);
      } else
#endif
      {
      v_tma_start = clock64();
#if ATTENTION_SPLIT_V_TMA
      begin_clock_trace_record(clock_trace, trace_slot_base + 3,
                               kClockTraceVTma, iter, pipe, role_warp_id, -1,
                               0, v_tma_start, clock_trace_base);
#else
      begin_clock_trace_record(clock_trace, trace_slot_base + 3,
                               kClockTraceVTma, iter, pipe, role_warp_id, -1,
                               -1, v_tma_start, clock_trace_base);
#endif
      }
    }
#endif
    if (local > 0) {
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_V_SERIAL || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_SERIAL
      if (pipe == 1) {
        mbarrier_wait(&v_ready[0], phase);
      }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_V_ISSUE || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
      if (pipe == 1) {
        cross_pipe_wait_at_least(&v_issue_gen[0],
                                 static_cast<uint32_t>(local + 1));
      }
#endif
#if ATTENTION_SPLIT_V_TMA
#if ATTENTION_SPLIT_V_H0_WITH_K_TMA
      if (!has_next_iter) {
        issue_v_tma_half_tile(v_map, v_smem[pipe], &v_ready[pipe],
                              global_v_tile, 0, lane0);
#if ATTENTION_CLOCK_TRACE
        if (trace_iter && lane0) {
          write_clock_trace_record(clock_trace,
                                   trace_slot_base + kClockTraceVTmaIssueSlot,
                                   kClockTraceVTmaIssue, iter, pipe,
                                   role_warp_id, -1, 0, v_tma_start, clock64(),
                                   clock_trace_base);
          const unsigned long long v_tma_h1_start = clock64();
          begin_clock_trace_record(clock_trace, trace_slot_base + 48,
                                   kClockTraceVTma, iter, pipe, role_warp_id,
                                   -1, 1, v_tma_h1_start, clock_trace_base);
          v_tma_start = v_tma_h1_start;
        }
#endif
      }
#else
      issue_v_tma_half_tile(v_map, v_smem[pipe], &v_ready[pipe], global_v_tile,
                            0, lane0);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter && lane0) {
        write_clock_trace_record(clock_trace,
                                 trace_slot_base + kClockTraceVTmaIssueSlot,
                                 kClockTraceVTmaIssue, iter, pipe,
                                 role_warp_id, -1, 0, v_tma_start, clock64(),
                                 clock_trace_base);
        const unsigned long long v_tma_h1_start = clock64();
        begin_clock_trace_record(clock_trace, trace_slot_base + 48,
                                 kClockTraceVTma, iter, pipe, role_warp_id, -1,
                                 1, v_tma_h1_start, clock_trace_base);
        v_tma_start = v_tma_h1_start;
      }
#endif
#endif
      issue_v_tma_half_tile(v_map, v_smem[pipe], &v_h1_ready[pipe],
                            global_v_tile, 1, lane0);
#else
      issue_v_tma_tile(v_map, v_smem[pipe], &v_ready[pipe], global_v_tile,
                       lane0);
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_V_ISSUE || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE
      if (pipe == 0 && lane0) {
        cross_pipe_signal(&v_issue_gen[0],
                          static_cast<uint32_t>(local + 1));
      }
#endif
    }
#if ATTENTION_CLOCK_TRACE
    if (trace_iter && local > 0) {
      write_clock_trace_record(clock_trace,
#if ATTENTION_SPLIT_V_TMA
                               trace_slot_base + 49,
                               kClockTraceVTmaIssue, iter, pipe, role_warp_id,
                               -1, 1, v_tma_start, clock64(),
                               clock_trace_base);
#else
                               trace_slot_base + kClockTraceVTmaIssueSlot,
                               kClockTraceVTmaIssue, iter, pipe, role_warp_id,
                               -1, -1, v_tma_start, clock64(),
                               clock_trace_base);
#endif
    }
#endif
  }
}

template <int kFixedKTiles>
__device__ ATTENTION_PIPE_ROLE_INLINE void attention_qk_pipe_role(
    uint32_t* q_smem,
    uint32_t* const (&k_smem)[kPipeCount],
    uint64_t* q_ready,
    uint64_t (&k_ready)[kPipeCount],
    uint64_t (&qk_done)[kPipeCount],
    uint64_t (&p_done)[kPipeCount],
    uint64_t (&s_h1_done)[kPipeCount],
    uint64_t (&pv_done)[kPipeCount],
    uint32_t (&qk_issue_gen)[kPipeCount],
    uint32_t* const (&s_smem)[kPipeCount],
    uint32_t* const (&v_smem)[kPipeCount],
    uint64_t (&v_ready)[kPipeCount],
    uint64_t (&v_h1_ready)[kPipeCount],
    const uint32_t (&p_taddr)[kPipeCount],
    const uint32_t (&o_taddr)[kPipeCount],
    int pipe,
    int loop_repeats,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    unsigned long long q_tma_start_shared,
    unsigned long long* k_tma_start_shared,
    unsigned int q_ready_phase,
#if ATTENTION_PERSISTENT_OVERLAP_EARLY
    uint64_t* qk_all_done_bar,
#endif
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
    bool qk_peeled,
#endif
    int lane) {
  const int role_warp_id = pipe;
  const bool lane0 = lane == 0;
#if !ATTENTION_CLOCK_TRACE
  (void)role_warp_id;
  (void)q_tma_start_shared;
  (void)k_tma_start_shared;
#endif
  const uint32_t idesc = make_qk_idesc();
  const uint32_t pv_idesc = make_qk_idesc() | (1u << 16);
#if ATTENTION_PERSISTENT_DESC_GEN
  QkDescGen q_desc{static_cast<uint32_t>(smem_ptr_u32(q_smem) >> 4)};
  QkDescGen k_desc{static_cast<uint32_t>(smem_ptr_u32(k_smem[pipe]) >> 4)};
  PvSDescGen pv_s_desc{static_cast<uint32_t>(smem_ptr_u32(s_smem[pipe]) >> 4)};
  PvVDescGen pv_v_desc{static_cast<uint32_t>(smem_ptr_u32(v_smem[pipe]) >> 4)};
#else
  uint64_t q_desc[8];
  uint64_t k_desc[8];
  uint64_t pv_s_desc[8];
  uint64_t pv_v_desc[8];
  if (lane0) {
    const uint32_t q_smem_addr16 = smem_ptr_u32(q_smem) >> 4;
    const uint32_t k_smem_addr16 = smem_ptr_u32(k_smem[pipe]) >> 4;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_sw128_major_k_smem_desc_addr16(q_smem_addr16, mma);
      k_desc[mma] = make_sw128_major_k_smem_desc_addr16(k_smem_addr16, mma);
    }
    {
      const uint32_t s_smem_addr16 = smem_ptr_u32(s_smem[pipe]) >> 4;
      const uint32_t v_smem_addr16 = smem_ptr_u32(v_smem[pipe]) >> 4;
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        pv_s_desc[mma] =
            make_s_smem_desc_addr16(s_smem_addr16 + static_cast<uint32_t>(mma) * (4096u >> 4));
        pv_v_desc[mma] = make_sw128_major_mn_smem_desc_addr16(v_smem_addr16, mma);
      }
    }
  }
#endif
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
  if (!qk_peeled)
#endif
  mbarrier_wait(q_ready, q_ready_phase);
#if ATTENTION_CLOCK_TRACE
  if (pipe == 0) {
    if (blockIdx.x == 0 && lane0 && clock_trace != nullptr) {
      const unsigned long long q_tma_end = clock64();
      const int q_tma_slot = clock_trace_iters * kClockTraceSlotsPerIter + 11;
      write_clock_trace_record(clock_trace, q_tma_slot, kClockTraceQTma, -1, -1,
                               role_warp_id, -1, -1, q_tma_start_shared, q_tma_end,
                               clock_trace_base);
    }
  }
#endif
  int iter = pipe;
  int local = 0;
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
  if (qk_peeled) {
    iter = pipe + kActivePipeStride;
    local = 1;
    PEEL_HB_SET(8 + pipe, 100u);
  } else
#endif
  if (iter < loop_repeats) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
#if ATTENTION_CLOCK_TRACE
    const int trace_idx = iter - clock_trace_start;
    const bool trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
        trace_idx < clock_trace_iters;
    const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
#endif
    mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      const unsigned long long k_tma_start =
          k_tma_start_shared[pipe * 2 + phase];
      if (k_tma_start != 0ull) {
        end_clock_trace_record(clock_trace, trace_slot_base + 1,
                               clock64(), clock_trace_base);
      }
    }
#endif
#if ATTENTION_PIPE1_QK_HEAD_DELAY_CYCLES > 0
    if (pipe == 1) {
      attention_clock_delay(ATTENTION_PIPE1_QK_HEAD_DELAY_CYCLES);
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_AFTER_PIPE0
    if (pipe == 1) {
      mbarrier_wait(&qk_done[0], phase);
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_ISSUE
    if (pipe == 1) {
      cross_pipe_wait_at_least(&qk_issue_gen[0], 1);
    }
#endif
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      unsigned long long qk_mma_start = 0ull;
      if (trace_iter) {
        qk_mma_start = clock64();
        begin_clock_trace_record(clock_trace, trace_slot_base + 2,
                                 kClockTraceQkMma, iter, pipe, role_warp_id, -1,
                                 -1, qk_mma_start, clock_trace_base);
      }
#endif
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc,
                            mma != 0);
      }
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_ISSUE
      if (pipe == 0) {
        cross_pipe_signal(&qk_issue_gen[0], 1);
      }
#endif
      tcgen05_commit(&qk_done[pipe]);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        write_clock_trace_record(clock_trace,
                                 trace_slot_base + kClockTraceQkMmaIssueSlot,
                                 kClockTraceQkMmaIssue, iter, pipe,
                                 role_warp_id, -1, -1, qk_mma_start, clock64(),
                                 clock_trace_base);
      }
#endif
    }
    iter += kActivePipeStride;
    ++local;
  }
  for (; iter < loop_repeats; iter += kActivePipeStride, ++local) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    const uint32_t prev_phase = static_cast<uint32_t>((local - 1) & 1);
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
    if (qk_peeled) PEEL_HB_SET(8 + pipe, 200u + static_cast<unsigned int>(local));
#endif
#if ATTENTION_CLOCK_TRACE
    const int trace_idx = iter - clock_trace_start;
    const bool trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
        trace_idx < clock_trace_iters;
    const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
#endif
#if ATTENTION_CLOCK_TRACE
    const int done_iter = iter - kActivePipeStride;
    const int done_trace_idx = done_iter - clock_trace_start;
#endif
    mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
    if (clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
        done_trace_idx >= 0 && done_trace_idx < clock_trace_iters) {
      end_clock_trace_record(clock_trace,
                             done_trace_idx * kClockTraceSlotsPerIter + 2,
                             clock64(), clock_trace_base);
    }
#endif
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      const unsigned long long k_tma_start =
          k_tma_start_shared[pipe * 2 + phase];
      if (k_tma_start != 0ull) {
        end_clock_trace_record(clock_trace, trace_slot_base + 1,
                               clock64(), clock_trace_base);
      }
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_AFTER_PIPE0
    if (pipe == 1) {
      mbarrier_wait(&qk_done[0], phase);
    }
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_ISSUE
    if (pipe == 1) {
      cross_pipe_wait_at_least(&qk_issue_gen[0],
                               static_cast<uint32_t>(local + 1));
    }
#endif
#if ATTENTION_CLOCK_TRACE
    const int pv_iter = iter - kActivePipeStride;
    const int pv_trace_idx = pv_iter - clock_trace_start;
    const bool pv_trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
        pv_trace_idx >= 0 && pv_trace_idx < clock_trace_iters;
    const int pv_trace_slot_base = pv_trace_idx * kClockTraceSlotsPerIter;
#endif
    mbarrier_wait(&p_done[pipe], prev_phase);
#if ATTENTION_CLOCK_TRACE
    unsigned long long qk_mma_start = 0ull;
#endif
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        qk_mma_start = clock64();
        begin_clock_trace_record(clock_trace, trace_slot_base + 2,
                                 kClockTraceQkMma, iter, pipe, role_warp_id, -1,
                                 -1, qk_mma_start, clock_trace_base);
      }
#endif
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc,
                            mma != 0);
      }
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_ISSUE
      if (pipe == 0) {
        cross_pipe_signal(&qk_issue_gen[0],
                          static_cast<uint32_t>(local + 1));
      }
#endif
    }
#if !(ATTENTION_SPLIT_V_TMA && ATTENTION_SKIP_V_H0_READY_WAIT_STEADY)
    mbarrier_wait(&v_ready[pipe], prev_phase);
#if ATTENTION_CLOCK_TRACE
    if (pv_trace_iter) {
      end_clock_trace_record(clock_trace, pv_trace_slot_base + 3, clock64(),
                             clock_trace_base);
    }
#endif
#endif
    if (lane0) {
#if ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER
#pragma unroll
      for (int mma = 0; mma < ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8;
           ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, local != 1 || mma != 0);
      }
      tcgen05_commit(&qk_done[pipe]);
#pragma unroll
      for (int mma = ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8;
           mma < kMmasPerTile / 2; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, true);
      }
#else
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile / 2; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, local != 1 || mma != 0);
      }
      tcgen05_commit(&qk_done[pipe]);
#endif
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        write_clock_trace_record(clock_trace,
                                 trace_slot_base + kClockTraceQkMmaIssueSlot,
                                 kClockTraceQkMmaIssue, iter, pipe,
                                 role_warp_id, -1, -1, qk_mma_start, clock64(),
                                 clock_trace_base);
      }
#endif
    }
    mbarrier_wait(&s_h1_done[pipe], prev_phase);
#if ATTENTION_SPLIT_V_TMA
#if !ATTENTION_SKIP_V_H1_READY_WAIT_STEADY
    mbarrier_wait(&v_h1_ready[pipe], prev_phase);
#if ATTENTION_CLOCK_TRACE
    if (pv_trace_iter) {
      end_clock_trace_record(clock_trace, pv_trace_slot_base + 48, clock64(),
                             clock_trace_base);
    }
#endif
#endif
#endif
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      const unsigned long long pv_h1_start =
          pv_trace_iter ? clock64() : 0ull;
      if (pv_trace_iter) {
        begin_clock_trace_record(clock_trace, pv_trace_slot_base + 4,
                                 kClockTracePvMma, pv_iter, pipe,
                                 role_warp_id, -1, 1, pv_h1_start,
                                 clock_trace_base);
      }
#endif
#pragma unroll
      for (int mma = kMmasPerTile / 2; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, true);
      }
#if ATTENTION_CLOCK_TRACE
      if (pv_trace_iter) {
        write_clock_trace_record(clock_trace, pv_trace_slot_base + 6,
                                 kClockTracePvMmaH1, pv_iter, pipe,
                                 role_warp_id, -1, 1, pv_h1_start, clock64(),
                                 clock_trace_base);
      }
#endif
      tcgen05_commit(&pv_done[pipe]);
    }
  }
  if (local > 0) {
    const uint32_t tail_phase = static_cast<uint32_t>((local - 1) & 1);
    mbarrier_wait(&qk_done[pipe], tail_phase);
#if ATTENTION_PERSISTENT_OVERLAP_EARLY
    if (lane0) mbarrier_arrive(qk_all_done_bar);
#endif
#if ATTENTION_CLOCK_TRACE
    const int done_iter = iter - kActivePipeStride;
    const int done_trace_idx = done_iter - clock_trace_start;
    if (clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
        done_trace_idx >= 0 && done_trace_idx < clock_trace_iters) {
      end_clock_trace_record(clock_trace,
                             done_trace_idx * kClockTraceSlotsPerIter + 2,
                             clock64(), clock_trace_base);
    }
#endif
#if ATTENTION_CLOCK_TRACE
    const int tail_iter = iter - kActivePipeStride;
    const int tail_trace_idx = tail_iter - clock_trace_start;
    const bool tail_trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
        tail_trace_idx >= 0 && tail_trace_idx < clock_trace_iters;
    const int tail_trace_slot_base = tail_trace_idx * kClockTraceSlotsPerIter;
#endif
    mbarrier_wait(&p_done[pipe], tail_phase);
#if !(ATTENTION_SPLIT_V_TMA && ATTENTION_SKIP_V_H0_READY_WAIT_TAIL)
    mbarrier_wait(&v_ready[pipe], tail_phase);
#endif
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_AFTER_PIPE0
    if (pipe == 1) {
      mbarrier_wait(&qk_done[0], tail_phase);
    }
#endif
#if ATTENTION_CLOCK_TRACE
#if !(ATTENTION_SPLIT_V_TMA && ATTENTION_SKIP_V_H0_READY_WAIT_TAIL)
    if (tail_trace_iter) {
      end_clock_trace_record(clock_trace, tail_trace_slot_base + 3, clock64(),
                             clock_trace_base);
    }
#endif
#endif
    if (lane0) {
#if ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER
#pragma unroll
      for (int mma = 0; mma < ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8;
           ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, local != 1 || mma != 0);
      }
      tcgen05_commit(&qk_done[pipe]);
#pragma unroll
      for (int mma = ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8;
           mma < kMmasPerTile / 2; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, true);
      }
#else
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile / 2; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, local != 1 || mma != 0);
      }
      tcgen05_commit(&qk_done[pipe]);
#endif
    }
    mbarrier_wait(&s_h1_done[pipe], tail_phase);
#if ATTENTION_SPLIT_V_TMA
#if !ATTENTION_SKIP_V_H1_READY_WAIT_TAIL
    mbarrier_wait(&v_h1_ready[pipe], tail_phase);
#if ATTENTION_CLOCK_TRACE
    if (tail_trace_iter) {
      end_clock_trace_record(clock_trace, tail_trace_slot_base + 48, clock64(),
                             clock_trace_base);
    }
#endif
#endif
#endif
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      const unsigned long long pv_h1_start =
          tail_trace_iter ? clock64() : 0ull;
      if (tail_trace_iter) {
        begin_clock_trace_record(clock_trace, tail_trace_slot_base + 4,
                                 kClockTracePvMma, tail_iter, pipe,
                                 role_warp_id, -1, 1, pv_h1_start,
                                 clock_trace_base);
      }
#endif
#pragma unroll
      for (int mma = kMmasPerTile / 2; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                            pv_idesc, true);
      }
#if ATTENTION_CLOCK_TRACE
      if (tail_trace_iter) {
        write_clock_trace_record(clock_trace, tail_trace_slot_base + 6,
                                 kClockTracePvMmaH1, tail_iter, pipe,
                                 role_warp_id, -1, 1, pv_h1_start, clock64(),
                                 clock_trace_base);
      }
#endif
      tcgen05_commit(&pv_done[pipe]);
    }
  }
}

#if ATTENTION_ROW_SUM_RARE_UPDATE
struct RowSumUpdateH0Result {
  float row_sum;
  float row_sum_reg;
  float row_max;
};

struct RowSumUpdateH1Result {
  float row_sum0;
  float row_sum1;
  float row_sum_reg;
  float row_max;
};

__device__ __noinline__ RowSumUpdateH0Result
attention_row_sum_update_h0_cold(uint32_t row_taddr,
                                 uint32_t* s_smem,
                                 uint64_t* p_done_barrier,
                                 uint32_t row_o_taddr,
                                 int pipe,
                                 int consumer_warp,
                                 int iter,
                                 bool trigger_update,
                                 float row_sum,
                                 float row_sum_reg,
                                 float row_max,
                                 float score_to_exp2_scale,
                                 bool do_row_sum,
                                 ClockTraceRecord* clock_trace,
                                 int clock_trace_iters,
                                 int clock_trace_start,
                                 unsigned long long clock_trace_base) {
  const float new_row_max =
      tcgen05_ld_x64_wait_row_max_scaled_nvcc(row_taddr, score_to_exp2_scale);
  const bool update = trigger_update && new_row_max > row_max;
  if (__any_sync(0xffffffffu, update)) {
    const float accum_scale =
        update ? exp2_approx_float_cpp(row_max - new_row_max) : 1.0f;
    if (do_row_sum) {
      row_sum_reg *= accum_scale;
      scale_tmem_x128_accum(row_o_taddr, accum_scale);
    }
    row_max = update ? new_row_max : row_max;
    row_sum = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
        row_taddr, s_smem, consumer_warp, 0, p_done_barrier, false,
        score_to_exp2_scale, row_max, clock_trace, clock_trace_iters,
        clock_trace_start, clock_trace_base, iter, pipe);
  }
  return {row_sum, row_sum_reg, row_max};
}

__device__ __noinline__ RowSumUpdateH1Result
attention_row_sum_update_h1_cold(uint32_t row_taddr,
                                 uint32_t* s_smem,
                                 uint64_t* p_done_barrier,
                                 uint32_t row_o_taddr,
                                 int pipe,
                                 int consumer_warp,
                                 int iter,
                                 bool trigger_update,
                                 float row_sum0,
                                 float row_sum1,
                                 float row_sum_reg,
                                 float row_max,
                                 float score_to_exp2_scale,
                                 bool do_row_sum,
                                 ClockTraceRecord* clock_trace,
                                 int clock_trace_iters,
                                 int clock_trace_start,
                                 unsigned long long clock_trace_base) {
  const float new_row_max =
      tcgen05_ld_x64_wait_row_max_scaled_nvcc(row_taddr, score_to_exp2_scale);
  const bool update = trigger_update && new_row_max > row_max;
  if (__any_sync(0xffffffffu, update)) {
    const float accum_scale =
        update ? exp2_approx_float_cpp(row_max - new_row_max) : 1.0f;
    if (do_row_sum) {
      row_sum_reg *= accum_scale;
      row_sum0 *= accum_scale;
      scale_tmem_x128_accum(row_o_taddr, accum_scale);
    }
    row_max = update ? new_row_max : row_max;
    row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
        row_taddr, s_smem, consumer_warp, 1, p_done_barrier, false,
        score_to_exp2_scale, row_max, clock_trace, clock_trace_iters,
        clock_trace_start, clock_trace_base, iter, pipe);
  }
  return {row_sum0, row_sum1, row_sum_reg, row_max};
}
#endif

__device__ ATTENTION_PIPE_ROLE_INLINE void attention_consumer_pipe_role(
    uint32_t* const (&s_smem)[kPipeCount],
    uint64_t (&qk_done)[kPipeCount],
    uint64_t (&p_done)[kPipeCount],
    uint64_t (&s_h1_done)[kPipeCount],
    float (&row_sum_partial)[kPipeCount][kTileM],
    float* row_max_scratch,
    const uint32_t (&p_taddr)[kPipeCount],
    const uint32_t (&o_taddr)[kPipeCount],
    int pipe,
    int consumer_warp,
    int loop_repeats,
    float score_to_exp2_scale,
    bool do_row_sum,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    int lane) {
  const int row = consumer_warp * 32 + lane;
  float row_sum_reg = 0.0f;
#if ATTENTION_FIRST_ITER_ROW_MAX_SHIFT || ATTENTION_ROW_MAX_ONLY
  float row_max_reg = -3.4028234663852886e+38f;
#endif
  int iter = pipe;
  int local = 0;
#if ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
#if ATTENTION_ROW_SUM_RARE_UPDATE
  const uint32_t row_o_taddr =
      o_taddr[pipe] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
#endif
#if ATTENTION_ROW_SUM_RARE_UPDATE
  const float row_sum_update_limit =
      static_cast<float>(ATTENTION_ROW_SUM_UPDATE_LIMIT);
#endif
  if (iter < loop_repeats) {
    mbarrier_wait(&qk_done[pipe], 0);
    const uint32_t row_taddr =
        p_taddr[pipe] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
#if ATTENTION_FIRST_ITER_COMPUTE_MAX
    row_max_reg = tcgen05_ld_x64_wait_row_max_scaled_nvcc(
        row_taddr, score_to_exp2_scale);
    row_max_reg =
        fmaxf(row_max_reg,
              tcgen05_ld_x64_wait_row_max_scaled_nvcc(row_taddr + 64u,
                                                      score_to_exp2_scale));
#else
    row_max_reg = 0.0f;
#endif
#if ATTENTION_FIRST_ITER_APPLY_SHIFT
    const float row_sum0 = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
        row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
        score_to_exp2_scale, row_max_reg, clock_trace, clock_trace_iters,
        clock_trace_start, clock_trace_base, iter, pipe);
#else
    const float row_sum0 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
#endif
#if ATTENTION_FIRST_ITER_APPLY_SHIFT
    const float row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
        row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe], true,
        score_to_exp2_scale, row_max_reg, clock_trace, clock_trace_iters,
        clock_trace_start, clock_trace_base, iter, pipe);
#else
    const float row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe], true,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
#endif
    if (lane == 0) mbarrier_arrive(&s_h1_done[pipe]);
    if (do_row_sum) row_sum_reg += row_sum0 + row_sum1;
    iter += kActivePipeStride;
    local = 1;
  }
#if ATTENTION_ROW_SUM_RARE_UPDATE && ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS > 0
#pragma unroll
  for (int prefix_check = 0;
       prefix_check < ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS; ++prefix_check) {
    if (iter < loop_repeats) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      mbarrier_wait(&qk_done[pipe], phase);
      const uint32_t row_taddr =
          p_taddr[pipe] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
      float row_sum0 = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
          row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
          score_to_exp2_scale, row_max_reg, clock_trace, clock_trace_iters,
          clock_trace_start, clock_trace_base, iter, pipe);
      const bool trigger_h0_update = !(row_sum0 <= row_sum_update_limit);
      if (__any_sync(0xffffffffu, trigger_h0_update)) {
        RowSumUpdateH0Result update_result = attention_row_sum_update_h0_cold(
            row_taddr, s_smem[pipe], &p_done[pipe], row_o_taddr, pipe,
            consumer_warp, iter, trigger_h0_update, row_sum0, row_sum_reg,
            row_max_reg, score_to_exp2_scale, do_row_sum, clock_trace,
            clock_trace_iters, clock_trace_start, clock_trace_base);
        row_sum0 = update_result.row_sum;
        row_sum_reg = update_result.row_sum_reg;
        row_max_reg = update_result.row_max;
      }
      float row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
          row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe], true,
          score_to_exp2_scale, row_max_reg, clock_trace, clock_trace_iters,
          clock_trace_start, clock_trace_base, iter, pipe);
      const bool trigger_h1_update = !(row_sum1 <= row_sum_update_limit);
      if (__any_sync(0xffffffffu, trigger_h1_update)) {
        RowSumUpdateH1Result update_result = attention_row_sum_update_h1_cold(
            row_taddr + 64u, s_smem[pipe], &p_done[pipe], row_o_taddr, pipe,
            consumer_warp, iter, trigger_h1_update, row_sum0, row_sum1,
            row_sum_reg, row_max_reg, score_to_exp2_scale, do_row_sum,
            clock_trace, clock_trace_iters, clock_trace_start,
            clock_trace_base);
        row_sum0 = update_result.row_sum0;
        row_sum1 = update_result.row_sum1;
        row_sum_reg = update_result.row_sum_reg;
        row_max_reg = update_result.row_max;
      }
      if (lane == 0) mbarrier_arrive(&s_h1_done[pipe]);
      if (do_row_sum) row_sum_reg += row_sum0 + row_sum1;
      iter += kActivePipeStride;
      ++local;
    }
  }
#endif
  for (; iter < loop_repeats; iter += kActivePipeStride, ++local) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    mbarrier_wait(&qk_done[pipe], phase);
    const uint32_t row_taddr =
        p_taddr[pipe] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
#if ATTENTION_FIRST_ITER_APPLY_SHIFT
    float row_sum0;
    float row_sum1;
    row_sum0 = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
        row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
        score_to_exp2_scale, row_max_reg, clock_trace, clock_trace_iters,
        clock_trace_start, clock_trace_base, iter, pipe);
#if ATTENTION_ROW_SUM_RARE_UPDATE && ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS == 0
    const bool trigger_h0_update = !(row_sum0 <= row_sum_update_limit);
    if (__any_sync(0xffffffffu, trigger_h0_update)) {
      RowSumUpdateH0Result update_result = attention_row_sum_update_h0_cold(
          row_taddr, s_smem[pipe], &p_done[pipe], row_o_taddr, pipe,
          consumer_warp, iter, trigger_h0_update, row_sum0, row_sum_reg,
          row_max_reg, score_to_exp2_scale, do_row_sum, clock_trace,
          clock_trace_iters, clock_trace_start, clock_trace_base);
      row_sum0 = update_result.row_sum;
      row_sum_reg = update_result.row_sum_reg;
      row_max_reg = update_result.row_max;
    }
#endif
#else
    const float row_sum0 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
#endif
#if ATTENTION_FIRST_ITER_APPLY_SHIFT
    row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
        row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe], true,
        score_to_exp2_scale, row_max_reg, clock_trace, clock_trace_iters,
        clock_trace_start, clock_trace_base, iter, pipe);
#if ATTENTION_ROW_SUM_RARE_UPDATE && ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS == 0
    const bool trigger_h1_update = !(row_sum1 <= row_sum_update_limit);
    if (__any_sync(0xffffffffu, trigger_h1_update)) {
      RowSumUpdateH1Result update_result = attention_row_sum_update_h1_cold(
          row_taddr + 64u, s_smem[pipe], &p_done[pipe], row_o_taddr, pipe,
          consumer_warp, iter, trigger_h1_update, row_sum0, row_sum1,
          row_sum_reg, row_max_reg, score_to_exp2_scale, do_row_sum,
          clock_trace, clock_trace_iters, clock_trace_start, clock_trace_base);
      row_sum0 = update_result.row_sum0;
      row_sum1 = update_result.row_sum1;
      row_sum_reg = update_result.row_sum_reg;
      row_max_reg = update_result.row_max;
    }
#endif
#else
    const float row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe], true,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
#endif
    if (lane == 0) mbarrier_arrive(&s_h1_done[pipe]);
    if (do_row_sum) row_sum_reg += row_sum0 + row_sum1;
  }
#else
  for (; iter < loop_repeats; iter += kActivePipeStride, ++local) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    mbarrier_wait(&qk_done[pipe], phase);
    const uint32_t row_taddr =
        p_taddr[pipe] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
#if ATTENTION_ROW_MAX_ONLY
    const PackStoreX64LoopResult h0_result =
        tcgen05_ld_x64_wait_pack_store_sum_max_half_nvcc(
            row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
            score_to_exp2_scale, clock_trace, clock_trace_iters,
            clock_trace_start, clock_trace_base, iter, pipe);
    row_max_reg = fmaxf(row_max_reg, h0_result.row_max);
    const float row_sum0 = h0_result.sum;
#else
    const float row_sum0 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
#endif
#if ATTENTION_ROW_MAX_ONLY
    const PackStoreX64LoopResult h1_result =
        tcgen05_ld_x64_wait_pack_store_sum_max_half_nvcc(
            row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe],
            true, score_to_exp2_scale, clock_trace, clock_trace_iters,
            clock_trace_start, clock_trace_base, iter, pipe);
    row_max_reg = fmaxf(row_max_reg, h1_result.row_max);
    const float row_sum1 = h1_result.sum;
#else
    const float row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe], true,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
#endif
    if (lane == 0) mbarrier_arrive(&s_h1_done[pipe]);
    if (do_row_sum) row_sum_reg += row_sum0 + row_sum1;
  }
#endif
#if ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
  if (do_row_sum && row_max_scratch != nullptr) {
    row_max_scratch[pipe * kTileM + row] = row_max_reg;
  }
#endif
  if (do_row_sum) {
    row_sum_partial[pipe][row] = row_sum_reg;
  }
#if ATTENTION_ROW_MAX_ONLY
  asm volatile("" :: "f"(row_max_reg) : "memory");
#endif
}

#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
template <int PIPE>
__device__ __forceinline__ void attention_issue_qk_peel(
    uint32_t* q_smem, uint32_t* const k_smem[kPipeCount],
    const uint32_t p_taddr[kPipeCount], uint64_t* q_ready,
    uint64_t k_ready[kPipeCount], uint64_t qk_peel_done[kPipeCount],
    unsigned int peel_q_phase, bool lane0) {
  mbarrier_wait(q_ready, peel_q_phase);
  mbarrier_wait(&k_ready[PIPE], 0u);
  if (lane0) {
#if !ATTENTION_PEEL_PREARM
    mbarrier_init(&qk_peel_done[PIPE], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
#if !ATTENTION_PEEL_NO_FENCE
    tcgen05_fence_after_thread_sync();
#endif
    const uint32_t peel_idesc = make_qk_idesc();
#if ATTENTION_PERSISTENT_DESC_GEN
    const QkDescGen qd{static_cast<uint32_t>(smem_ptr_u32(q_smem) >> 4)};
    const QkDescGen kd{static_cast<uint32_t>(smem_ptr_u32(k_smem[PIPE]) >> 4)};
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      tcgen05_mma_bf16_ss(p_taddr[PIPE], qd[mma], kd[mma], peel_idesc, mma != 0);
    }
#else
    const uint32_t q16 = smem_ptr_u32(q_smem) >> 4;
    const uint32_t k16 = smem_ptr_u32(k_smem[PIPE]) >> 4;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      tcgen05_mma_bf16_ss(p_taddr[PIPE],
                          make_sw128_major_k_smem_desc_addr16(q16, mma),
                          make_sw128_major_k_smem_desc_addr16(k16, mma), peel_idesc,
                          mma != 0);
    }
#endif
    tcgen05_commit(&qk_peel_done[PIPE]);
#if !ATTENTION_PEEL_NO_FENCE
    tcgen05_fence_before_thread_sync();
#endif
  }
}

template <int PIPE>
__device__ __forceinline__ void attention_peel_issue_mma_only(
    uint32_t* q_smem, uint32_t* const k_smem[kPipeCount],
    const uint32_t p_taddr[kPipeCount], uint64_t* q_ready,
    uint64_t k_ready[kPipeCount], uint64_t qk_peel_done[kPipeCount],
    unsigned int peel_q_phase, bool lane0) {
  mbarrier_wait(q_ready, peel_q_phase);
  mbarrier_wait(&k_ready[PIPE], 0u);
  if (lane0) {
#if !ATTENTION_PEEL_PREARM
    mbarrier_init(&qk_peel_done[PIPE], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
#if !ATTENTION_PEEL_NO_FENCE
    tcgen05_fence_after_thread_sync();
#endif
    const uint32_t peel_idesc = make_qk_idesc();
#if ATTENTION_PERSISTENT_DESC_GEN
    const QkDescGen qd{static_cast<uint32_t>(smem_ptr_u32(q_smem) >> 4)};
    const QkDescGen kd{static_cast<uint32_t>(smem_ptr_u32(k_smem[PIPE]) >> 4)};
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      tcgen05_mma_bf16_ss(p_taddr[PIPE], qd[mma], kd[mma], peel_idesc, mma != 0);
    }
#else
    const uint32_t q16 = smem_ptr_u32(q_smem) >> 4;
    const uint32_t k16 = smem_ptr_u32(k_smem[PIPE]) >> 4;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      tcgen05_mma_bf16_ss(p_taddr[PIPE],
                          make_sw128_major_k_smem_desc_addr16(q16, mma),
                          make_sw128_major_k_smem_desc_addr16(k16, mma), peel_idesc,
                          mma != 0);
    }
#endif
  }
}

template <int PIPE>
__device__ __forceinline__ void attention_peel_commit_only(
    uint64_t qk_peel_done[kPipeCount], bool lane0) {
  if (lane0) {
    tcgen05_commit(&qk_peel_done[PIPE]);
#if !ATTENTION_PEEL_NO_FENCE
    tcgen05_fence_before_thread_sync();
#endif
  }
}
#endif

template <int kFixedRepeats = 0, int kFixedKTiles = 0>
__global__ __launch_bounds__(kMainThreads, 1)
void qk_tma_mma_ld_kernel(const __grid_constant__ CUtensorMap q_map,
                          const __grid_constant__ CUtensorMap k_map,
                          const __grid_constant__ CUtensorMap v_map,
                          const __grid_constant__ CUtensorMap o_map,
                          int repeats,
                          int k_tiles,
                          float score_to_exp2_scale,
                          void* __restrict__ output,
                          int total_tiles
#if ATTENTION_CLOCK_TRACE
                          ,
                          ClockTraceRecord* __restrict__ clock_trace,
                          int clock_trace_iters,
                          int clock_trace_start
#endif
                          ) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)v_map;
  (void)o_map;
  (void)repeats;
  (void)k_tiles;
  (void)score_to_exp2_scale;
  (void)output;
  (void)total_tiles;
#if ATTENTION_CLOCK_TRACE
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
#endif
#else
  const int loop_repeats = kFixedRepeats > 0 ? kFixedRepeats : repeats;
  const int loop_k_tiles = kFixedKTiles > 0 ? kFixedKTiles : k_tiles;
  extern __shared__ uint32_t smem_raw[];
#if !ATTENTION_PERSISTENT
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    k_smem[p] = q_smem + (1 + p) * kTileWords;
  }
  uint32_t* v_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    v_smem[p] = q_smem + (1 + kKBufferTileCount + p) * kTileWords;
  }
  uint32_t* s_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    s_smem[p] = q_smem + (1 + kKBufferTileCount + kVBufferCount + p) * kTileWords;
  }
#endif

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[kPipeCount];
  __shared__ uint64_t qk_done[kPipeCount];
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
  __shared__ uint64_t qk_peel_done[kPipeCount];
#endif
  __shared__ uint64_t p_done[kPipeCount];
  __shared__ uint64_t s_h1_done[kPipeCount];
  __shared__ uint64_t v_ready[kPipeCount];
  __shared__ uint64_t v_h1_ready[kPipeCount];
  __shared__ uint64_t pv_done[kPipeCount];
#if ATTENTION_PERSISTENT_OVERLAP_EARLY
  __shared__ uint64_t qk_all_done;
#endif
  __shared__ uint32_t k_issue_gen[kPipeCount];
  __shared__ uint32_t v_issue_gen[kPipeCount];
  __shared__ uint32_t qk_issue_gen[kPipeCount];
  __shared__ uint64_t tma_head_marker;
  __shared__ float row_sum_partial[kPipeCount][kTileM];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
#if ATTENTION_CLOCK_TRACE
  __shared__ unsigned long long clock_trace_base_shared;
  __shared__ unsigned long long q_tma_start_shared;
  __shared__ unsigned long long k_tma_start_shared[kPipeCount * 2];
  __shared__ unsigned long long tail_total_start_shared;
  __shared__ unsigned long long tma_store_start_shared;
#else
  ClockTraceRecord* clock_trace = nullptr;
  const int clock_trace_iters = 0;
  const int clock_trace_start = 0;
  const unsigned long long clock_trace_base_shared = 0ull;
  const unsigned long long q_tma_start_shared = 0ull;
  unsigned long long* k_tma_start_shared = nullptr;
#endif

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;

  if (warp_id == 0 || warp_id == 1) {
    setmaxnreg_dec_qk();
  } else if (warp_id == 2 || warp_id == 3) {
    setmaxnreg_dec_tma();
  } else {
    setmaxnreg_inc_consumer();
  }

#if ATTENTION_PERSISTENT
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();
  for (int tile = blockIdx.x; tile < total_tiles;
       tile += static_cast<int>(gridDim.x)) {
  uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  asm volatile("" : "+l"(smem_addr));
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    k_smem[p] = q_smem + (1 + p) * kTileWords;
  }
  uint32_t* v_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    v_smem[p] = q_smem + (1 + kKBufferTileCount + p) * kTileWords;
  }
  uint32_t* s_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    s_smem[p] = q_smem + (1 + kKBufferTileCount + kVBufferCount + p) * kTileWords;
  }
  volatile uint32_t* tmem_base_reload = &tmem_base_shared;
  const uint32_t tmem_base = *tmem_base_reload;
  const uint32_t p_taddr[kPipeCount] = {tmem_base, tmem_base + 128u};
  const uint32_t o_taddr[kPipeCount] = {tmem_base + 256u, tmem_base + 384u};
#endif

#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
  const bool qk_peeled =
#if ATTENTION_PEEL_ROLE_NOSKIP
      false;
#else
      output != nullptr && tile != static_cast<int>(blockIdx.x) &&
      loop_repeats >= kActivePipeStride;
#endif
#else
  const bool qk_peeled = false;
#endif
  (void)qk_peeled;

#if ATTENTION_PERSISTENT_OVERLAP
  if (tile == static_cast<int>(blockIdx.x) && threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
#pragma unroll
    for (int p = 0; p < kPipeCount; ++p) {
      mbarrier_init(&k_ready[p], 1);
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL && !ATTENTION_PEEL_NO_INITONCE
      mbarrier_init(&qk_peel_done[p], 1);
#endif
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#endif
  if (threadIdx.x == 0) {
#if !ATTENTION_PERSISTENT_OVERLAP
    mbarrier_init(&q_ready, 1);
#endif
#if ATTENTION_PERSISTENT_OVERLAP_EARLY
    mbarrier_init(&qk_all_done, 2);
#endif
#if ATTENTION_PIPE1_TMA_HEAD_MARKER
    mbarrier_init(&tma_head_marker, 1);
#endif
#pragma unroll
    for (int p = 0; p < kPipeCount; ++p) {
#if !ATTENTION_PERSISTENT_OVERLAP
      mbarrier_init(&k_ready[p], 1);
#endif
      mbarrier_init(&qk_done[p], 1);
      mbarrier_init(&p_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&s_h1_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&v_ready[p], 1);
      mbarrier_init(&v_h1_ready[p], 1);
      mbarrier_init(&pv_done[p], 1);
#if ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_K_ISSUE || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_V_ISSUE || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_TMA_KV_ISSUE || \
      ATTENTION_CROSS_PIPE_PHASE == ATTENTION_CROSS_PHASE_QK_ISSUE
      k_issue_gen[p] = 0;
      v_issue_gen[p] = 0;
      qk_issue_gen[p] = 0;
#endif
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (output != nullptr && loop_repeats < kPipeCount) {
    for (int i = threadIdx.x; i < kPipeCount * kTileM; i += blockDim.x) {
      reinterpret_cast<float*>(row_sum_partial)[i] = 0.0f;
    }
  }
  __syncthreads();
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
  if (qk_peeled && warp_id == 0) {
    PEEL_HB_SET(12, 1u);
    mbarrier_wait(&qk_peel_done[0], 0u);
    PEEL_HB_SET(12, 2u);
    if (lane0) mbarrier_arrive(&qk_done[0]);
#if ATTENTION_PEEL_PREARM
    if (lane0) {
      mbarrier_init(&qk_peel_done[0], 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
#endif
    PEEL_HB_SET(12, 3u);
  } else if (qk_peeled && warp_id == 1) {
    PEEL_HB_SET(13, 1u);
    mbarrier_wait(&qk_peel_done[1], 0u);
    PEEL_HB_SET(13, 2u);
    if (lane0) mbarrier_arrive(&qk_done[1]);
#if ATTENTION_PEEL_PREARM
    if (lane0) {
      mbarrier_init(&qk_peel_done[1], 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
#endif
    PEEL_HB_SET(13, 3u);
  }
#endif
#if ATTENTION_CLOCK_TRACE
  if (threadIdx.x == 0) {
    clock_trace_base_shared = clock64();
#pragma unroll
    for (int i = 0; i < kPipeCount * 2; ++i) {
      k_tma_start_shared[i] = 0ull;
    }
  }
  __syncthreads();
#endif
  const unsigned long long clock_trace_base = clock_trace_base_shared;
#if !ATTENTION_PERSISTENT
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[kPipeCount] = {tmem_base, tmem_base + 128u};
  const uint32_t o_taddr[kPipeCount] = {tmem_base + 256u, tmem_base + 384u};
  const int tile = static_cast<int>(blockIdx.x);
#endif
  float* row_max_scratch =
      output != nullptr
          ? reinterpret_cast<float*>(output) +
                static_cast<size_t>(tile) * kTileWords
          : nullptr;
  const int q_contig_row = tile * kTileM;
  const int kv_tile_base =
      kv_tile_base_for_block<kFixedKTiles>(tile, loop_k_tiles);
#if ATTENTION_PERSISTENT_OVERLAP
  const int tile_local_idx =
      (tile - static_cast<int>(blockIdx.x)) / static_cast<int>(gridDim.x);
  const unsigned int q_ready_phase = static_cast<unsigned int>(tile_local_idx & 1);
  const bool k_prefetched = ATTENTION_PERSISTENT_OVERLAP_PREFETCH &&
      output != nullptr && tile != static_cast<int>(blockIdx.x);
#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
  const bool wait_prev_store =
      output != nullptr && tile != static_cast<int>(blockIdx.x);
#endif
#else
  const unsigned int q_ready_phase = 0u;
  const bool k_prefetched = false;
#endif

  if (warp_id == 0
#if ATTENTION_PERSISTENT_OVERLAP && ATTENTION_PERSISTENT_OVERLAP_PREFETCH
      && (output == nullptr || tile == static_cast<int>(blockIdx.x))
#endif
  ) {
#if ATTENTION_CLOCK_TRACE
    if (lane0) q_tma_start_shared = clock64();
#endif
    mbarrier_expect_tx(&q_ready, kTileBytes);
    if (lane0) {
      const uint32_t q_smem_addr = smem_ptr_u32(q_smem);
      tma_load_2d(&q_map, q_smem_addr, &q_ready, 0, q_contig_row);
      tma_load_2d(&q_map, q_smem_addr + kTileBytes / 2, &q_ready, 32, q_contig_row);
    }
  }

  if (warp_id >= kConsumerBaseWarp &&
      warp_id < kConsumerBaseWarp + kActiveConsumerPipeCount * kConsumerWarpsPerPipe) {
    const int pipe = (warp_id - kConsumerBaseWarp) / kConsumerWarpsPerPipe;
    const int consumer_slot =
        (warp_id - kConsumerBaseWarp) - pipe * kConsumerWarpsPerPipe;
    const int consumer_warp = consumer_slot;
    attention_consumer_pipe_role(
        s_smem, qk_done, p_done, s_h1_done, row_sum_partial,
        row_max_scratch, p_taddr, o_taddr, pipe, consumer_warp, loop_repeats,
        score_to_exp2_scale,
        output != nullptr, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, lane);
  }

  if (warp_id == 2 || warp_id == 3) {
    const int pipe = warp_id - 2;
    attention_pv_pipe_role<kFixedKTiles>(
        &k_map, &v_map, k_smem, v_smem, k_ready, v_ready, v_h1_ready,
        qk_done, pv_done,
        k_issue_gen, v_issue_gen, &tma_head_marker,
        k_tma_start_shared, pipe, loop_repeats, loop_k_tiles, kv_tile_base, clock_trace,
        clock_trace_iters, clock_trace_start, clock_trace_base, k_prefetched,
#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
        wait_prev_store,
#endif
#if ATTENTION_PEEL_HB
        qk_peeled,
#endif
        lane);
  }

	  if (warp_id == 0 || warp_id == 1) {
	    const int pipe = warp_id;
	    attention_qk_pipe_role<kFixedKTiles>(
	        q_smem, k_smem, &q_ready, k_ready, qk_done, p_done, s_h1_done, pv_done,
		        qk_issue_gen,
	        s_smem, v_smem, v_ready, v_h1_ready, p_taddr, o_taddr, pipe,
        loop_repeats, clock_trace,
        clock_trace_iters, clock_trace_start, clock_trace_base,
        q_tma_start_shared, k_tma_start_shared, q_ready_phase,
#if ATTENTION_PERSISTENT_OVERLAP_EARLY
        &qk_all_done,
#endif
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL
        qk_peeled,
#endif
        lane);
  }

#if ATTENTION_CLOCK_TRACE
  const bool trace_cta = clock_trace != nullptr && blockIdx.x == 0;
  const int trace_extra_base = clock_trace_iters * kClockTraceSlotsPerIter;
#else
  const bool trace_cta = false;
  const int trace_extra_base = 0;
#endif

  if (output != nullptr) {
#if ATTENTION_CLOCK_TRACE
    const unsigned long long tail_total_start =
        trace_cta && threadIdx.x == 0 ? clock64() : 0ull;
    if (trace_cta && threadIdx.x == 0) {
      tail_total_start_shared = tail_total_start;
    }
    const unsigned long long tail_wait_start =
        trace_cta && threadIdx.x == 0 ? tail_total_start : 0ull;
#endif
    const int pipe0_local_count =
        (loop_repeats + 1) / 2;
    const int pipe1_local_count = loop_repeats / 2;
#if ATTENTION_PERSISTENT_OVERLAP_EARLY
    if (warp_id == 2 || warp_id == 3) {
      const int next_tile = tile + static_cast<int>(gridDim.x);
      if (next_tile < total_tiles) {
        mbarrier_wait(&qk_all_done, 0);
        if (warp_id == 2) {
          mbarrier_expect_tx(&q_ready, kTileBytes);
          if (lane0) {
            const uint32_t q_smem_addr = smem_ptr_u32(q_smem);
            const int next_q_row = next_tile * kTileM;
            tma_load_2d(&q_map, q_smem_addr, &q_ready, 0, next_q_row);
            tma_load_2d(&q_map, q_smem_addr + kTileBytes / 2, &q_ready, 32, next_q_row);
          }
        }
        const int pf_pipe = warp_id - 2;
        const int next_kv_base =
            kv_tile_base_for_block<kFixedKTiles>(next_tile, loop_k_tiles);
        const int pf_k_tile =
            local_k_tile_for_iter<kFixedKTiles>(pf_pipe, loop_k_tiles);
        issue_k_tma_tile(&k_map, k_smem[pf_pipe], &k_ready[pf_pipe],
                         next_kv_base + pf_k_tile, lane0);
      }
    }
#endif
    if (pipe0_local_count > 0) {
      mbarrier_wait(&pv_done[0], static_cast<uint32_t>((pipe0_local_count - 1) & 1));
#if ATTENTION_CLOCK_TRACE
      if (trace_cta && threadIdx.x == 0) {
        const int done_iter = (pipe0_local_count - 1) * kActivePipeStride;
        if (done_iter >= clock_trace_start &&
            done_iter < clock_trace_start + clock_trace_iters) {
          end_clock_trace_record(
              clock_trace,
              (done_iter - clock_trace_start) * kClockTraceSlotsPerIter + 4,
              clock64(), clock_trace_base);
        }
      }
#endif
    }
    if (pipe1_local_count > 0) {
      mbarrier_wait(&pv_done[1], static_cast<uint32_t>((pipe1_local_count - 1) & 1));
#if ATTENTION_CLOCK_TRACE
      if (trace_cta && threadIdx.x == 0) {
        const int done_iter = (pipe1_local_count - 1) * kActivePipeStride + 1;
        if (done_iter >= clock_trace_start &&
            done_iter < clock_trace_start + clock_trace_iters) {
          end_clock_trace_record(
              clock_trace,
              (done_iter - clock_trace_start) * kClockTraceSlotsPerIter + 4,
              clock64(), clock_trace_base);
        }
      }
#endif
    }
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL && ATTENTION_PEEL_DURING_DRAIN && \
    !ATTENTION_PEEL_AFTER_SYNC
#if ATTENTION_PEEL_DD_PRESYNC
    __syncthreads();
#endif
#if ATTENTION_PEEL_DD_SINGLE_ISSUER
    if (warp_id == 0 && loop_repeats >= kActivePipeStride) {
      const int peel_next_tile = tile + static_cast<int>(gridDim.x);
      if (peel_next_tile < total_tiles) {
        const unsigned int peel_q_phase =
            static_cast<unsigned int>((tile_local_idx + 1) & 1);
        PEEL_HB_SET(0, 3u);
        attention_issue_qk_peel<0>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                   qk_peel_done, peel_q_phase, lane0);
        PEEL_HB_SET(0, 4u);
        PEEL_HB_SET(1, 3u);
        attention_issue_qk_peel<1>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                   qk_peel_done, peel_q_phase, lane0);
        PEEL_HB_SET(1, 4u);
      }
    }
#elif ATTENTION_PEEL_DD_SERIAL_COMMIT
    if ((warp_id == 0 || warp_id == 1) && loop_repeats >= kActivePipeStride) {
      const int peel_next_tile = tile + static_cast<int>(gridDim.x);
      if (peel_next_tile < total_tiles) {
        const unsigned int peel_q_phase =
            static_cast<unsigned int>((tile_local_idx + 1) & 1);
        PEEL_HB_SET(warp_id, 3u);
        if (warp_id == 0) {
          attention_peel_issue_mma_only<0>(q_smem, k_smem, p_taddr, &q_ready,
                                           k_ready, qk_peel_done, peel_q_phase, lane0);
        } else {
          attention_peel_issue_mma_only<1>(q_smem, k_smem, p_taddr, &q_ready,
                                           k_ready, qk_peel_done, peel_q_phase, lane0);
        }
        asm volatile("bar.sync 6, 64;" ::: "memory");
        if (warp_id == 0) attention_peel_commit_only<0>(qk_peel_done, lane0);
        asm volatile("bar.sync 6, 64;" ::: "memory");
        if (warp_id == 1) attention_peel_commit_only<1>(qk_peel_done, lane0);
        PEEL_HB_SET(warp_id, 4u);
      }
    }
#else
    if ((warp_id == 0 || warp_id == 1) && loop_repeats >= kActivePipeStride) {
      const int peel_next_tile = tile + static_cast<int>(gridDim.x);
      if (peel_next_tile < total_tiles) {
        const unsigned int peel_q_phase =
            static_cast<unsigned int>((tile_local_idx + 1) & 1);
        PEEL_HB_SET(warp_id, 3u);
        if (warp_id == 0) {
          attention_issue_qk_peel<0>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0);
        } else {
          attention_issue_qk_peel<1>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0);
        }
        PEEL_HB_SET(warp_id, 4u);
      }
    }
#endif
#endif
#if ATTENTION_CLOCK_TRACE
    if (trace_cta && threadIdx.x == 0) {
      const unsigned long long tail_wait_end = clock64();
      write_clock_trace_record(clock_trace, trace_extra_base, kClockTraceTailWait,
                               loop_repeats, -1, 0, -1, -1, tail_wait_start,
                               tail_wait_end, clock_trace_base);
    }
#endif
#if ATTENTION_PERSISTENT_OVERLAP_PREFETCH && !ATTENTION_PERSISTENT_OVERLAP_EARLY
    {
      const int next_tile = tile + static_cast<int>(gridDim.x);
      if (next_tile < total_tiles) {
        if (warp_id == 2) {
          mbarrier_expect_tx(&q_ready, kTileBytes);
          if (lane0) {
            const uint32_t q_smem_addr = smem_ptr_u32(q_smem);
            const int next_q_row = next_tile * kTileM;
            tma_load_2d(&q_map, q_smem_addr, &q_ready, 0, next_q_row);
            tma_load_2d(&q_map, q_smem_addr + kTileBytes / 2, &q_ready, 32, next_q_row);
          }
        }
        if (warp_id == 2 || warp_id == 3) {
          const int pf_pipe = warp_id - 2;
          const int next_kv_base =
              kv_tile_base_for_block<kFixedKTiles>(next_tile, loop_k_tiles);
          const int pf_k_tile =
              local_k_tile_for_iter<kFixedKTiles>(pf_pipe, loop_k_tiles);
          issue_k_tma_tile(&k_map, k_smem[pf_pipe], &k_ready[pf_pipe],
                           next_kv_base + pf_k_tile, lane0);
        }
      }
    }
#endif
    __syncthreads();
#if ATTENTION_PERSISTENT_OVERLAP_O_IN_V
    uint32_t* output_bf16_smem = v_smem[0];
#elif ATTENTION_EPILOGUE_O_IN_S_SMEM
    uint32_t* output_bf16_smem = s_smem[0];
#else
    uint32_t* output_bf16_smem = reinterpret_cast<uint32_t*>(q_smem);
#endif
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL && ATTENTION_PEEL_DURING_DRAIN && \
    ATTENTION_PEEL_AFTER_SYNC
    if ((warp_id == 0 || warp_id == 1) && loop_repeats >= kActivePipeStride) {
      const int peel_next_tile = tile + static_cast<int>(gridDim.x);
      if (peel_next_tile < total_tiles) {
        const unsigned int peel_q_phase =
            static_cast<unsigned int>((tile_local_idx + 1) & 1);
        PEEL_HB_SET(warp_id, 3u);
        if (warp_id == 0) {
          attention_issue_qk_peel<0>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0);
        } else {
          attention_issue_qk_peel<1>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0);
        }
        PEEL_HB_SET(warp_id, 4u);
      }
    }
#endif
    const bool epilogue_warp =
        warp_id >= kConsumerBaseWarp &&
        warp_id < kConsumerBaseWarp + kPipeCount * kConsumerWarpsPerPipe;
    const bool trace_epilogue =
        trace_cta && lane0 && warp_id >= kConsumerBaseWarp &&
        warp_id < kConsumerBaseWarp + kConsumerWarpsPerPipe;
    const unsigned long long epilogue_start =
        trace_epilogue ? clock64() : 0ull;
    if (pipe0_local_count > 0 && epilogue_warp) {
      const int epilogue_slot = warp_id - kConsumerBaseWarp;
      const int consumer_warp = epilogue_slot & (kConsumerWarpsPerPipe - 1);
      const int consumer_half = epilogue_slot / kConsumerWarpsPerPipe;
      const int row = consumer_warp * 32 + lane;
#if ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE
      const float row_max0 = row_max_scratch[row];
      const float row_max1 =
          pipe1_local_count > 0 ? row_max_scratch[kTileM + row] : row_max0;
      const float common_row_max = fmaxf(row_max0, row_max1);
      const float pipe0_scale = exp2_approx_float_cpp(row_max0 - common_row_max);
      const float pipe1_scale =
          pipe1_local_count > 0
              ? exp2_approx_float_cpp(row_max1 - common_row_max)
              : 0.0f;
      const float denom = row_sum_partial[0][row] * pipe0_scale +
                          row_sum_partial[1][row] * pipe1_scale;
#else
      const float denom = row_sum_partial[0][row] + row_sum_partial[1][row];
#endif
      const float inv_sum = denom != 0.0f ? 1.0f / denom : 0.0f;
      const uint32_t row_taddr0 =
          o_taddr[0] + (static_cast<uint32_t>(consumer_warp * 32) << 16) +
          static_cast<uint32_t>(consumer_half * 64);
      const uint32_t row_taddr1 =
          o_taddr[1] + (static_cast<uint32_t>(consumer_warp * 32) << 16) +
          static_cast<uint32_t>(consumer_half * 64);
      uint32_t* row_dst =
          output_bf16_smem + static_cast<size_t>(row) * (kTileN / 2) +
          consumer_half * (kTileN / 4);
#if ATTENTION_EPILOGUE_CHUNK_COLS == 16
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
        const uint32_t chunk_offset = static_cast<uint32_t>(chunk * 16);
        if (pipe1_local_count > 0) {
#if ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE
          store_tmem_x16_pair_scale_norm_bf16_smem(
              row_taddr0 + chunk_offset, row_taddr1 + chunk_offset,
              row_dst + chunk * 8, pipe0_scale, pipe1_scale, inv_sum);
#else
          store_tmem_x16_pair_norm_bf16_smem(
              row_taddr0 + chunk_offset, row_taddr1 + chunk_offset,
              row_dst + chunk * 8, inv_sum);
#endif
        } else {
          store_tmem_x16_norm_bf16_smem(row_taddr0 + chunk_offset,
                                        row_dst + chunk * 8, inv_sum);
        }
      }
#elif ATTENTION_EPILOGUE_CHUNK_COLS == 32
      if (pipe1_local_count > 0) {
#if ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE
        store_tmem_x32_pair_scale_norm_bf16_smem(
            row_taddr0, row_taddr1, row_dst, pipe0_scale, pipe1_scale,
            inv_sum);
        store_tmem_x32_pair_scale_norm_bf16_smem(
            row_taddr0 + 32u, row_taddr1 + 32u, row_dst + 16,
            pipe0_scale, pipe1_scale, inv_sum);
#else
        store_tmem_x32_pair_norm_bf16_smem(row_taddr0, row_taddr1, row_dst,
                                           inv_sum);
        store_tmem_x32_pair_norm_bf16_smem(row_taddr0 + 32u, row_taddr1 + 32u,
                                           row_dst + 16, inv_sum);
#endif
      } else {
        store_tmem_x32_norm_bf16_smem(row_taddr0, row_dst, inv_sum);
        store_tmem_x32_norm_bf16_smem(row_taddr0 + 32u, row_dst + 16,
                                      inv_sum);
      }
#else
#error "ATTENTION_EPILOGUE_CHUNK_COLS must be 16 or 32"
#endif
    }
    if (trace_epilogue) {
      const unsigned long long epilogue_end = clock64();
      const int consumer_warp = warp_id - kConsumerBaseWarp;
      write_clock_trace_record(clock_trace, trace_extra_base + 1 + consumer_warp,
                               kClockTraceTmemDrain, loop_repeats, -1, warp_id,
                               consumer_warp, -1, epilogue_start, epilogue_end,
                               clock_trace_base);
      write_clock_trace_record(clock_trace, trace_extra_base + 5 + consumer_warp,
                               kClockTracePackNorm, loop_repeats, -1, warp_id,
                               consumer_warp, -1, epilogue_start, epilogue_end,
                               clock_trace_base);
    }
    tma_store_fence();
    __syncthreads();
#if ATTENTION_PEEL_ISO
#if ATTENTION_ISO_ARRAY
    __shared__ uint64_t iso_bar[kPipeCount];
#ifndef ATTENTION_ISO_ARRAY_IDX
#define ATTENTION_ISO_ARRAY_IDX warp_id
#endif
    uint64_t* const iso_b = &iso_bar[ATTENTION_ISO_ARRAY_IDX];
#else
    __shared__ uint64_t iso_bar;
    uint64_t* const iso_b = &iso_bar;
#endif
    if (warp_id == 0 && loop_repeats >= kActivePipeStride) {
      const int next_tile = tile + static_cast<int>(gridDim.x);
      if (next_tile < total_tiles && lane0) {
        PEEL_HB_SET(0, 1u);
        const unsigned int iso_q_phase =
            static_cast<unsigned int>((tile_local_idx + 1) & 1);
        mbarrier_wait(&q_ready, iso_q_phase);
        PEEL_HB_SET(0, 2u);
        mbarrier_wait(&k_ready[0], 0u);
        PEEL_HB_SET(0, 3u);
        mbarrier_init(iso_b, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#if !ATTENTION_PEEL_NO_FENCE
        tcgen05_fence_after_thread_sync();
#endif
        const uint32_t iso_idesc = make_qk_idesc();
        const QkDescGen iso_q_desc{
            static_cast<uint32_t>(smem_ptr_u32(q_smem) >> 4)};
        const QkDescGen iso_k_desc{
            static_cast<uint32_t>(smem_ptr_u32(k_smem[0]) >> 4)};
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[0], iso_q_desc[mma], iso_k_desc[mma],
                              iso_idesc, mma != 0);
        }
        tcgen05_commit(iso_b);
#if !ATTENTION_PEEL_NO_FENCE
        tcgen05_fence_before_thread_sync();
#endif
        PEEL_HB_SET(0, 4u);
        mbarrier_wait(iso_b, 0u);
        PEEL_HB_SET(0, 5u);
      }
    }
#endif
#if ATTENTION_PERSISTENT_OVERLAP_QK_PEEL && !ATTENTION_PEEL_DURING_DRAIN
    if ((warp_id == 0 || warp_id == 1) && loop_repeats >= kActivePipeStride) {
      const int next_tile = tile + static_cast<int>(gridDim.x);
      if (next_tile < total_tiles) {
        const unsigned int peel_q_phase =
            static_cast<unsigned int>((tile_local_idx + 1) & 1);
        PEEL_HB_SET(warp_id, 3u);
        if (warp_id == 0) {
          attention_issue_qk_peel<0>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0);
        } else {
          attention_issue_qk_peel<1>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0);
        }
        PEEL_HB_SET(warp_id, 4u);
      }
    }
#endif
#if ATTENTION_CLOCK_TRACE
    if (trace_cta && threadIdx.x == 0) {
      tma_store_start_shared = clock64();
    }
#endif
#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
    if (lane0 && warp_id == 2) {
#else
    if (lane0 && warp_id == 0) {
#endif
      tma_store_4d(&o_map, smem_ptr_u32(output_bf16_smem), 0, 0,
                   tile, 0);
      tma_store_commit_group();
    }
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
  }
  __syncthreads();

#if !ATTENTION_PERSISTENT
  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
#if !ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
  if (output != nullptr && lane0 && warp_id == 0) {
    tma_store_wait_group_read();
  }
#endif
#if ATTENTION_CLOCK_TRACE
  if (output != nullptr && trace_cta && threadIdx.x == 0) {
    const unsigned long long store_end = clock64();
    write_clock_trace_record(clock_trace, trace_extra_base + 9,
                             kClockTraceGlobalStore, loop_repeats, -1, 0, -1, -1,
                             tma_store_start_shared, store_end,
                             clock_trace_base);
    write_clock_trace_record(clock_trace, trace_extra_base + 10,
                             kClockTraceTailTotal, loop_repeats, -1, 0, -1, -1,
                             tail_total_start_shared, store_end,
                             clock_trace_base);
  }
#endif
#if ATTENTION_PERSISTENT
  __syncthreads();
  }

#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
  if (output != nullptr && warp_id == 2 && lane0) {
    tma_store_wait_group_read();
  }
  __syncthreads();
#endif

  volatile uint32_t* tmem_base_reload_final = &tmem_base_shared;
  const uint32_t tmem_base_final = *tmem_base_reload_final;
  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base_final);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
#endif
}
