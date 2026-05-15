// Core CUDA kernels for the fused attention benchmark.
// Included by main.cu so templated kernels remain in the same translation unit.

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
#else
  ClockTraceRecord* clock_trace = nullptr;
  const int clock_trace_iters = 0;
  const int clock_trace_start = 0;
  const unsigned long long clock_trace_base_shared = 0ull;
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
  for (int i = threadIdx.x; i < kPipeCount * kTileM; i += blockDim.x) {
    reinterpret_cast<float*>(row_sum_partial)[i] = 0.0f;
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
    const bool do_row_sum = output != nullptr;
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
      if (pipe == 0) {
      } else {
      }
      const uint32_t row_taddr = p_taddr[pipe] +
                                 (static_cast<uint32_t>(consumer_warp * 32) << 16);
      const float row_sum0 =
          tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
              row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
              score_to_exp2_scale, clock_trace, clock_trace_iters,
              clock_trace_start, clock_trace_base, iter, pipe);
      if (lane0) {
        mbarrier_arrive(&s_ready[pipe][0]);
      }
      if (pipe == 0) {
      } else {
      }
      const float row_sum1 =
          tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
              row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe],
              true, score_to_exp2_scale, clock_trace, clock_trace_iters,
              clock_trace_start, clock_trace_base, iter, pipe);
      if (do_row_sum) row_sum_reg += row_sum0 + row_sum1;
      if (lane0) {
        mbarrier_arrive(&s_ready[pipe][1]);
      }
    }
    if (do_row_sum) {
      row_sum_partial[pipe][row] = row_sum_reg;
    }
  }

  if (warp_id == 2 || warp_id == 3) {
    const int pipe = warp_id - 2;
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
      issue_v_tma_tile(&v_map, v_smem[pipe], &v_ready[pipe], global_v_tile,
                       lane0);
      mbarrier_wait(&s_ready[pipe][0], phase);
      mbarrier_wait(&v_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        const unsigned long long v_tma_end = clock64();
        write_clock_trace_record(clock_trace, trace_slot_base + 3,
                                 kClockTraceVTma, iter, pipe, warp_id, -1, -1,
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
                                   kClockTracePvMma, iter, pipe, warp_id, -1,
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
                                   kClockTracePvMmaH0, iter, pipe, warp_id, -1,
                                   0, pv_h0_start, clock64(),
                                   clock_trace_base);
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
                                   kClockTracePvMmaH1, iter, pipe, warp_id, -1,
                                   1, pv_h1_start, clock64(),
                                   clock_trace_base);
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

  if (warp_id == 0 || warp_id == 1) {
    const int pipe = warp_id;
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
    mbarrier_wait(&q_ready, 0);
#if ATTENTION_CLOCK_TRACE
    if (blockIdx.x == 0 && lane0 && warp_id == 0 && clock_trace != nullptr) {
      const unsigned long long q_tma_end = clock64();
      const int q_tma_slot = clock_trace_iters * kClockTraceSlotsPerIter + 11;
      write_clock_trace_record(clock_trace, q_tma_slot, kClockTraceQTma, -1, -1,
                               warp_id, -1, -1, q_tma_start_shared, q_tma_end,
                               clock_trace_base);
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
      issue_k_tma_tile(&k_map, k_smem[pipe], &k_ready[pipe], global_k_tile,
                       lane0);
      mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        const unsigned long long k_tma_end = clock64();
        write_clock_trace_record(clock_trace, trace_slot_base + 1,
                                 kClockTraceKTma, iter, pipe, warp_id, -1, -1,
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
                                   kClockTraceQkMma, iter, pipe, warp_id, -1,
                                   -1, clock64(), clock_trace_base);
        }
#endif
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc, mma != 0);
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
      issue_k_tma_tile(&k_map, k_smem[pipe], &k_ready[pipe], global_k_tile,
                       lane0);
      mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        const unsigned long long k_tma_end = clock64();
        write_clock_trace_record(clock_trace, trace_slot_base + 1,
                                 kClockTraceKTma, iter, pipe, warp_id, -1, -1,
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
                                   kClockTraceQkMma, iter, pipe, warp_id, -1,
                                   -1, clock64(), clock_trace_base);
        }
#endif
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc, mma != 0);
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

  if (output != nullptr) {
#if ATTENTION_CLOCK_TRACE
    const bool trace_cta = clock_trace != nullptr && blockIdx.x == 0;
    const int trace_extra_base = clock_trace_iters * kClockTraceSlotsPerIter;
    const unsigned long long tail_total_start =
        trace_cta && threadIdx.x == 0 ? clock64() : 0ull;
    const unsigned long long tail_wait_start =
        trace_cta && threadIdx.x == 0 ? tail_total_start : 0ull;
#else
    const bool trace_cta = false;
    const int trace_extra_base = 0;
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
    float* output_smem = reinterpret_cast<float*>(q_smem);
    uint32_t* output_bf16_smem =
        reinterpret_cast<uint32_t*>(output_smem + kTileBf16Elems);
    const bool trace_drain =
        trace_cta && lane0 && warp_id >= kConsumerBaseWarp &&
        warp_id < kConsumerBaseWarp + kConsumerWarpsPerPipe;
    const unsigned long long drain_start = trace_drain ? clock64() : 0ull;
    if (pipe0_local_count > 0 && pipe1_local_count > 0 && warp_id >= kConsumerBaseWarp &&
        warp_id < kConsumerBaseWarp + kConsumerWarpsPerPipe) {
      const int consumer_warp = warp_id - kConsumerBaseWarp;
      const uint32_t row_taddr0 =
          o_taddr[0] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
      const uint32_t row_taddr1 =
          o_taddr[1] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
      store_tmem_x64_accum_output_smem(row_taddr0, output_smem, consumer_warp, 0,
                                       false);
      store_tmem_x64_accum_output_smem(row_taddr0 + 64u, output_smem, consumer_warp,
                                       1, false);
      store_tmem_x64_accum_output_smem(row_taddr1, output_smem, consumer_warp, 0,
                                       true);
      store_tmem_x64_accum_output_smem(row_taddr1 + 64u, output_smem, consumer_warp,
                                       1, true);
    } else if (pipe0_local_count > 0 && pipe1_local_count == 0 &&
               warp_id >= kConsumerBaseWarp &&
               warp_id < kConsumerBaseWarp + kConsumerWarpsPerPipe) {
      const int consumer_warp = warp_id - kConsumerBaseWarp;
      const uint32_t row_taddr =
          o_taddr[0] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
      store_tmem_x64_accum_output_smem(row_taddr, output_smem, consumer_warp, 0,
                                       false);
      store_tmem_x64_accum_output_smem(row_taddr + 64u, output_smem, consumer_warp,
                                       1, false);
    }
    if (trace_drain) {
      const unsigned long long drain_end = clock64();
      const int consumer_warp = warp_id - kConsumerBaseWarp;
      write_clock_trace_record(clock_trace, trace_extra_base + 1 + consumer_warp,
                               kClockTraceTmemDrain, loop_repeats, -1, warp_id,
                               consumer_warp, -1, drain_start, drain_end,
                               clock_trace_base);
    }
    __syncthreads();
    const bool trace_pack =
        trace_cta && lane0 && warp_id >= kConsumerBaseWarp &&
        warp_id < kConsumerBaseWarp + kConsumerWarpsPerPipe;
    const unsigned long long pack_start = trace_pack ? clock64() : 0ull;
    if (warp_id >= kConsumerBaseWarp && warp_id < kConsumerBaseWarp + kConsumerWarpsPerPipe) {
      const int consumer_warp = warp_id - kConsumerBaseWarp;
      const int row = consumer_warp * 32 + lane;
      const float denom = row_sum_partial[0][row] + row_sum_partial[1][row];
      const float inv_sum = denom != 0.0f ? 1.0f / denom : 0.0f;
      const float* row_src = output_smem + static_cast<size_t>(row) * kTileN;
      uint32_t* row_dst = output_bf16_smem + static_cast<size_t>(row) * (kTileN / 2);
#pragma unroll
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const int col = col_pair * 2;
        row_dst[col_pair] =
            pack_bf16_pair_device(row_src[col] * inv_sum, row_src[col + 1] * inv_sum);
      }
    }
    if (trace_pack) {
      const unsigned long long pack_end = clock64();
      const int consumer_warp = warp_id - kConsumerBaseWarp;
      write_clock_trace_record(clock_trace, trace_extra_base + 5 + consumer_warp,
                               kClockTracePackNorm, loop_repeats, -1, warp_id,
                               consumer_warp, -1, pack_start, pack_end,
                               clock_trace_base);
    }
    __syncthreads();
    const unsigned long long store_start =
        trace_cta && threadIdx.x == 0 ? clock64() : 0ull;
    uint32_t* output_words = reinterpret_cast<uint32_t*>(output);
    const size_t output_tile_base = static_cast<size_t>(blockIdx.x) * kTileWords;
    for (int word = threadIdx.x; word < kTileWords; word += blockDim.x) {
      output_words[output_tile_base + word] = output_bf16_smem[word];
    }
    if (trace_cta && threadIdx.x == 0) {
      const unsigned long long store_end = clock64();
      write_clock_trace_record(clock_trace, trace_extra_base + 9,
                               kClockTraceGlobalStore, loop_repeats, -1, 0, -1, -1,
                               store_start, store_end, clock_trace_base);
    }
#if ATTENTION_CLOCK_TRACE
    __syncthreads();
    if (trace_cta && threadIdx.x == 0) {
      const unsigned long long tail_total_end = clock64();
      write_clock_trace_record(clock_trace, trace_extra_base + 10,
                               kClockTraceTailTotal, loop_repeats, -1, 0, -1, -1,
                               tail_total_start, tail_total_end,
                               clock_trace_base);
    }
#endif
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}


