// Core CUDA kernels for the fused attention benchmark.
// Included by main.cu so templated kernels remain in the same translation unit.

#ifndef ATTENTION_EPILOGUE_CHUNK_COLS
#define ATTENTION_EPILOGUE_CHUNK_COLS 16
#endif

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
__device__ __forceinline__ void attention_pv_pipe_role(
    const CUtensorMap* v_map,
    uint32_t* const (&v_smem)[kPipeCount],
    uint32_t* const (&s_smem)[kPipeCount],
    uint64_t (&v_ready)[kPipeCount],
    uint64_t (&s_ready)[kPipeCount][2],
    uint64_t (&pv_done)[kPipeCount],
    const uint32_t (&o_taddr)[kPipeCount],
    int pipe,
    int loop_repeats,
    int loop_k_tiles,
    int kv_tile_base,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    int lane) {
  const int role_warp_id = 2 + pipe;
  const bool lane0 = lane == 0;
#if !ATTENTION_CLOCK_TRACE
  (void)role_warp_id;
#endif
  const uint32_t idesc = make_qk_idesc() | (1u << 16);
  uint64_t s_desc[8];
  uint64_t v_desc[8];
  if (lane0) {
    const uint32_t s_smem_addr16 = smem_ptr_u32(s_smem[pipe]) >> 4;
    const uint32_t v_smem_addr16 = smem_ptr_u32(v_smem[pipe]) >> 4;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      s_desc[mma] =
          make_s_smem_desc_addr16(s_smem_addr16 + static_cast<uint32_t>(mma) * (4096u >> 4));
      v_desc[mma] = make_sw128_major_mn_smem_desc_addr16(v_smem_addr16, mma);
    }
  }
  int iter = pipe;
  int local = 0;
  for (; iter < loop_repeats; iter += kActivePipeStride, ++local) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    const int global_v_tile =
        kv_tile_base + local_k_tile_for_iter<kFixedKTiles>(iter, loop_k_tiles);
#if ATTENTION_CLOCK_TRACE
    const int trace_idx = iter - clock_trace_start;
    const bool trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
        trace_idx < clock_trace_iters;
    const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
    unsigned long long v_tma_start = 0ull;
#endif
    if (local > 0) {
      mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
    }
#if ATTENTION_CLOCK_TRACE
    if (trace_iter && lane0) v_tma_start = clock64();
#endif
    issue_v_tma_tile(v_map, v_smem[pipe], &v_ready[pipe], global_v_tile,
                     lane0);
    mbarrier_wait(&s_ready[pipe][0], phase);
    mbarrier_wait(&v_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      const unsigned long long v_tma_end = clock64();
      write_clock_trace_record(clock_trace, trace_slot_base + 3,
                               kClockTraceVTma, iter, pipe, role_warp_id, -1, -1,
                               v_tma_start, v_tma_end, clock_trace_base);
    }
#endif
    if (pipe == 0) {
      if (local > 0) {
        mbarrier_wait(&pv_done[1], static_cast<uint32_t>((local - 1) & 1));
      }
    } else {
      mbarrier_wait(&pv_done[0], phase);
    }
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        begin_clock_trace_record(clock_trace, trace_slot_base + 4,
                                 kClockTracePvMma, iter, pipe, role_warp_id, -1,
                                 -1, clock64(), clock_trace_base);
      }
      const unsigned long long pv_h0_start =
          trace_iter ? clock64() : 0ull;
#endif
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile / 2; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], s_desc[mma], v_desc[mma], idesc,
                            local != 0 || mma != 0);
      }
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        write_clock_trace_record(clock_trace, trace_slot_base + 5,
                                 kClockTracePvMmaH0, iter, pipe, role_warp_id, -1,
                                 0, pv_h0_start, clock64(), clock_trace_base);
      }
#endif
    }
    mbarrier_wait(&s_ready[pipe][1], phase);
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      const unsigned long long pv_h1_start =
          trace_iter ? clock64() : 0ull;
#endif
#pragma unroll
      for (int mma = kMmasPerTile / 2; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(o_taddr[pipe], s_desc[mma], v_desc[mma], idesc,
                            true);
      }
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        write_clock_trace_record(clock_trace, trace_slot_base + 6,
                                 kClockTracePvMmaH1, iter, pipe, role_warp_id, -1,
                                 1, pv_h1_start, clock64(), clock_trace_base);
      }
#endif
      tcgen05_commit(&pv_done[pipe]);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        mbarrier_wait(&pv_done[pipe], phase);
        end_clock_trace_record(clock_trace, trace_slot_base + 4, clock64(),
                               clock_trace_base);
      }
#endif
    }
  }
}

template <int kFixedKTiles>
__device__ __forceinline__ void attention_qk_pipe_role(
    const CUtensorMap* k_map,
    uint32_t* q_smem,
    uint32_t* const (&k_smem)[kPipeCount],
    uint64_t* q_ready,
    uint64_t (&k_ready)[kPipeCount],
    uint64_t (&qk_done)[kPipeCount],
    uint64_t (&p_done)[kPipeCount],
    const uint32_t (&p_taddr)[kPipeCount],
    int pipe,
    int loop_repeats,
    int loop_k_tiles,
    int kv_tile_base,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    unsigned long long q_tma_start_shared,
    int lane) {
  const int role_warp_id = pipe;
  const bool lane0 = lane == 0;
#if !ATTENTION_CLOCK_TRACE
  (void)role_warp_id;
  (void)q_tma_start_shared;
#endif
  const uint32_t idesc = make_qk_idesc();
  uint64_t q_desc[8];
  uint64_t k_desc[8];
  if (lane0) {
    const uint32_t q_smem_addr16 = smem_ptr_u32(q_smem) >> 4;
    const uint32_t k_smem_addr16 = smem_ptr_u32(k_smem[pipe]) >> 4;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_sw128_major_k_smem_desc_addr16(q_smem_addr16, mma);
      k_desc[mma] = make_sw128_major_k_smem_desc_addr16(k_smem_addr16, mma);
    }
  }
  mbarrier_wait(q_ready, 0);
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
  if (iter < loop_repeats) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    const int k_tile = local_k_tile_for_iter<kFixedKTiles>(iter, loop_k_tiles);
    const int global_k_tile = kv_tile_base + k_tile;
#if ATTENTION_CLOCK_TRACE
    const int trace_idx = iter - clock_trace_start;
    const bool trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
        trace_idx < clock_trace_iters;
    const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
    unsigned long long k_tma_start = 0ull;
#endif
#if ATTENTION_CLOCK_TRACE
    if (trace_iter && lane0) k_tma_start = clock64();
#endif
    issue_k_tma_tile(k_map, k_smem[pipe], &k_ready[pipe], global_k_tile,
                     lane0);
    mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      const unsigned long long k_tma_end = clock64();
      write_clock_trace_record(clock_trace, trace_slot_base + 1,
                               kClockTraceKTma, iter, pipe, role_warp_id, -1, -1,
                               k_tma_start, k_tma_end, clock_trace_base);
    }
#endif
    if (pipe != 0) {
      mbarrier_wait(&qk_done[0], phase);
    }
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        begin_clock_trace_record(clock_trace, trace_slot_base + 2,
                                 kClockTraceQkMma, iter, pipe, role_warp_id, -1,
                                 -1, clock64(), clock_trace_base);
      }
#endif
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc,
                            mma != 0);
      }
      tcgen05_commit(&qk_done[pipe]);
    }
    iter += kActivePipeStride;
    ++local;
  }
  for (; iter < loop_repeats; iter += kActivePipeStride, ++local) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    const uint32_t prev_phase = static_cast<uint32_t>((local - 1) & 1);
    const int k_tile = local_k_tile_for_iter<kFixedKTiles>(iter, loop_k_tiles);
    const int global_k_tile = kv_tile_base + k_tile;
#if ATTENTION_CLOCK_TRACE
    const int trace_idx = iter - clock_trace_start;
    const bool trace_iter =
        clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
        trace_idx < clock_trace_iters;
    const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
    unsigned long long k_tma_start = 0ull;
#endif
    mbarrier_wait(&qk_done[pipe], prev_phase);
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
    if (trace_iter && lane0) k_tma_start = clock64();
#endif
    issue_k_tma_tile(k_map, k_smem[pipe], &k_ready[pipe], global_k_tile,
                     lane0);
    mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
    if (trace_iter) {
      const unsigned long long k_tma_end = clock64();
      write_clock_trace_record(clock_trace, trace_slot_base + 1,
                               kClockTraceKTma, iter, pipe, role_warp_id, -1, -1,
                               k_tma_start, k_tma_end, clock_trace_base);
    }
#endif
    mbarrier_wait(&p_done[pipe], prev_phase);
    if (pipe == 0) {
      mbarrier_wait(&qk_done[1], prev_phase);
    } else {
      mbarrier_wait(&qk_done[0], phase);
    }
    if (lane0) {
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        begin_clock_trace_record(clock_trace, trace_slot_base + 2,
                                 kClockTraceQkMma, iter, pipe, role_warp_id, -1,
                                 -1, clock64(), clock_trace_base);
      }
#endif
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc,
                            mma != 0);
      }
      tcgen05_commit(&qk_done[pipe]);
    }
  }
#if ATTENTION_CLOCK_TRACE
  if (local > 0) {
    mbarrier_wait(&qk_done[pipe], static_cast<uint32_t>((local - 1) & 1));
    const int done_iter = iter - kActivePipeStride;
    const int done_trace_idx = done_iter - clock_trace_start;
    if (clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
        done_trace_idx >= 0 && done_trace_idx < clock_trace_iters) {
      end_clock_trace_record(clock_trace,
                             done_trace_idx * kClockTraceSlotsPerIter + 2,
                             clock64(), clock_trace_base);
    }
  }
#endif
}

__device__ __forceinline__ void attention_consumer_pipe_role(
    uint32_t* const (&s_smem)[kPipeCount],
    uint64_t (&qk_done)[kPipeCount],
    uint64_t (&p_done)[kPipeCount],
    uint64_t (&s_ready)[kPipeCount][2],
    uint64_t (&pv_done)[kPipeCount],
    float (&row_sum_partial)[kPipeCount][kTileM],
    const uint32_t (&p_taddr)[kPipeCount],
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
  const bool lane0 = lane == 0;
  const int row = consumer_warp * 32 + lane;
  float row_sum_reg = 0.0f;
  int iter = pipe;
  int local = 0;
  for (; iter < loop_repeats; iter += kActivePipeStride, ++local) {
    const uint32_t phase = static_cast<uint32_t>(local & 1);
    mbarrier_wait(&qk_done[pipe], phase);
    if (local > 0) {
      mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
    }
    const uint32_t row_taddr =
        p_taddr[pipe] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
    const float row_sum0 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
    if (lane0) {
      mbarrier_arrive(&s_ready[pipe][0]);
    }
    const float row_sum1 = tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
        row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe], true,
        score_to_exp2_scale, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, iter, pipe);
    if (do_row_sum) row_sum_reg += row_sum0 + row_sum1;
    if (lane0) {
      mbarrier_arrive(&s_ready[pipe][1]);
    }
  }
  if (do_row_sum) {
    row_sum_partial[pipe][row] = row_sum_reg;
  }
}

template <int kFixedRepeats = 0, int kFixedKTiles = 0>
__global__ __launch_bounds__(kMainThreads, 1)
void qk_tma_mma_ld_kernel(const __grid_constant__ CUtensorMap q_map,
                          const __grid_constant__ CUtensorMap k_map,
                          const __grid_constant__ CUtensorMap v_map,
                          const __grid_constant__ CUtensorMap o_map,
                          int repeats,
                          int k_tiles,
                          float score_to_exp2_scale,
                          void* __restrict__ output
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
#if ATTENTION_CLOCK_TRACE
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
#endif
#else
  const int loop_repeats = kFixedRepeats > 0 ? kFixedRepeats : repeats;
  const int loop_k_tiles = kFixedKTiles > 0 ? kFixedKTiles : k_tiles;
  extern __shared__ uint32_t smem_raw[];
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

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[kPipeCount];
  __shared__ uint64_t qk_done[kPipeCount];
  __shared__ uint64_t p_done[kPipeCount];
  __shared__ uint64_t v_ready[kPipeCount];
  __shared__ uint64_t s_ready[kPipeCount][2];
  __shared__ uint64_t pv_done[kPipeCount];
  __shared__ float row_sum_partial[kPipeCount][kTileM];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
#if ATTENTION_CLOCK_TRACE
  __shared__ unsigned long long clock_trace_base_shared;
  __shared__ unsigned long long q_tma_start_shared;
  __shared__ unsigned long long tail_total_start_shared;
  __shared__ unsigned long long tma_store_start_shared;
#else
  ClockTraceRecord* clock_trace = nullptr;
  const int clock_trace_iters = 0;
  const int clock_trace_start = 0;
  const unsigned long long clock_trace_base_shared = 0ull;
  const unsigned long long q_tma_start_shared = 0ull;
#endif

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;

  if (warp_id < kProducerWarpCount) {
    setmaxnreg_dec_producer();
  } else {
    setmaxnreg_inc_consumer();
  }

  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
#pragma unroll
    for (int p = 0; p < kPipeCount; ++p) {
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&qk_done[p], 1);
      mbarrier_init(&p_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&v_ready[p], 1);
      mbarrier_init(&s_ready[p][0], kConsumerWarpsPerPipe);
      mbarrier_init(&s_ready[p][1], kConsumerWarpsPerPipe);
      mbarrier_init(&pv_done[p], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (output != nullptr && loop_repeats < kPipeCount) {
    for (int i = threadIdx.x; i < kPipeCount * kTileM; i += blockDim.x) {
      reinterpret_cast<float*>(row_sum_partial)[i] = 0.0f;
    }
  }
  __syncthreads();
#if ATTENTION_CLOCK_TRACE
  if (threadIdx.x == 0) {
    clock_trace_base_shared = clock64();
  }
  __syncthreads();
#endif
  const unsigned long long clock_trace_base = clock_trace_base_shared;
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[kPipeCount] = {tmem_base, tmem_base + 128u};
  const uint32_t o_taddr[kPipeCount] = {tmem_base + 256u, tmem_base + 384u};
  const int q_contig_row = static_cast<int>(blockIdx.x) * kTileM;
  const int kv_tile_base =
      kv_tile_base_for_block<kFixedKTiles>(static_cast<int>(blockIdx.x),
                                           loop_k_tiles);

  if (warp_id == 0) {
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
        s_smem, qk_done, p_done, s_ready, pv_done, row_sum_partial, p_taddr,
        pipe, consumer_warp, loop_repeats, score_to_exp2_scale,
        output != nullptr, clock_trace, clock_trace_iters, clock_trace_start,
        clock_trace_base, lane);
  }

  if (warp_id == 2 || warp_id == 3) {
    const int pipe = warp_id - 2;
    attention_pv_pipe_role<kFixedKTiles>(
        &v_map, v_smem, s_smem, v_ready, s_ready, pv_done, o_taddr, pipe,
        loop_repeats, loop_k_tiles, kv_tile_base, clock_trace,
        clock_trace_iters, clock_trace_start, clock_trace_base, lane);
  }

  if (warp_id == 0 || warp_id == 1) {
    const int pipe = warp_id;
    attention_qk_pipe_role<kFixedKTiles>(
        &k_map, q_smem, k_smem, &q_ready, k_ready, qk_done, p_done,
        p_taddr, pipe, loop_repeats, loop_k_tiles, kv_tile_base, clock_trace,
        clock_trace_iters, clock_trace_start, clock_trace_base,
        q_tma_start_shared, lane);
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
    if (pipe0_local_count > 0) {
      mbarrier_wait(&pv_done[0], static_cast<uint32_t>((pipe0_local_count - 1) & 1));
    }
    if (pipe1_local_count > 0) {
      mbarrier_wait(&pv_done[1], static_cast<uint32_t>((pipe1_local_count - 1) & 1));
    }
#if ATTENTION_CLOCK_TRACE
    if (trace_cta && threadIdx.x == 0) {
      const unsigned long long tail_wait_end = clock64();
      write_clock_trace_record(clock_trace, trace_extra_base, kClockTraceTailWait,
                               loop_repeats, -1, 0, -1, -1, tail_wait_start,
                               tail_wait_end, clock_trace_base);
    }
#endif
    __syncthreads();
    uint32_t* output_bf16_smem = reinterpret_cast<uint32_t*>(q_smem);
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
      const float denom = row_sum_partial[0][row] + row_sum_partial[1][row];
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
          store_tmem_x16_pair_norm_bf16_smem(
              row_taddr0 + chunk_offset, row_taddr1 + chunk_offset,
              row_dst + chunk * 8, inv_sum);
        } else {
          store_tmem_x16_norm_bf16_smem(row_taddr0 + chunk_offset,
                                        row_dst + chunk * 8, inv_sum);
        }
      }
#elif ATTENTION_EPILOGUE_CHUNK_COLS == 32
      if (pipe1_local_count > 0) {
        store_tmem_x32_pair_norm_bf16_smem(row_taddr0, row_taddr1, row_dst,
                                           inv_sum);
        store_tmem_x32_pair_norm_bf16_smem(row_taddr0 + 32u, row_taddr1 + 32u,
                                           row_dst + 16, inv_sum);
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
#if ATTENTION_CLOCK_TRACE
    if (trace_cta && threadIdx.x == 0) {
      tma_store_start_shared = clock64();
    }
#endif
    if (lane0 && warp_id == 0) {
      tma_store_4d(&o_map, smem_ptr_u32(output_bf16_smem), 0, 0,
                   static_cast<int>(blockIdx.x), 0);
      tma_store_commit_group();
    }
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
  if (output != nullptr && lane0 && warp_id == 0) {
    tma_store_wait_group_read();
  }
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
#endif
}
