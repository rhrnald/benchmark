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

// Continuous flatten pipeline (15_FLATTEN_IMPL.md): all roles own their own
// cross-tile loop; no outer per-tile loop, no CTA boundary __syncthreads, so
// QK(t+1) overlaps drain(t). Only the live benchmark config is supported (see
// 15 §0). Off => D1 (2ce1d8a) path is byte-for-byte intact.
#ifndef ATTENTION_CONTINUOUS_FLAT
#define ATTENTION_CONTINUOUS_FLAT 0
#endif
// Deadlock locator: when on, every mbarrier_wait in the flat block becomes a
// timed spin that printf's (line, warp, phase) if it stalls > ~0.5s, then breaks
// (false-success) so the kernel limps to exit and flushes the printf buffer.
#ifndef ATTENTION_FLAT_DEBUG
#define ATTENTION_FLAT_DEBUG 0
#endif
// NOTE: the producer waits v_ready/v_h1_ready on EVERY PV in the flat path.
// Under TMA_PUSH the V arrives early enough that these waits are ~free (push ==
// base perf), and skipping them (D1-style timing / sparse GUARD) caused
// nondeterministic GPU faults — dead experiments removed 2026-07-02 (see
// markdown/20 §2). Build with -DATTENTION_SKIP_V_TMA_EXPECT_TX=0 so the V
// barriers are actually armed.
// F1 (19_DIAGNOSIS_SCHEDULE_NOT_CONTENTION.md §3): rewrite the flat TMA loop in
// the master PUSH-style choreography. Body gl issues K(gl+1) + EARLY Vh0(gl) as
// soon as qk_done(gl) fires (master steady order, attention_pv_pipe_role
// 796->882->936); only Vh1(gl) (+ the final iter's Vh0) waits pv_done(gl-1)
// (master 977->1033->1051). The default pull loop instead reaches its K(gl)/V(gl)
// issues only after pv_done(gl-1) in TMA program order, so K/Vh0 land ~7 PV-MMAs
// (~1,300cyc) later every iter (k_ready 800->1765, v_ready ~1000 newly exposed)
// = the flat -12%. Compute barriers' arrive/commit are untouched (16 §9.0
// doctrine); qk_done is consumed once per body in order (bodies 0..G-2), so
// running parity holds (15 §6b). The early Vh0 runs master's exact timing race
// (TMA flight > PV h0 tail MMAs) — D1-proven, ck bit-identical.
#ifndef ATTENTION_FLAT_TMA_PUSH
#define ATTENTION_FLAT_TMA_PUSH 0
#endif
#if ATTENTION_FLAT_TMA_PUSH && !ATTENTION_CONTINUOUS_FLAT
#error "ATTENTION_FLAT_TMA_PUSH requires ATTENTION_CONTINUOUS_FLAT"
#endif
// G4 seam de-convoy (19 §5-G4). At each tile seam (producer body L==0) the flat
// loop issues QK(t+1,i0) BEFORE PV(t,last), so tile t's last PV — and the
// pv_done confirm + pv_tile_done arrive the drain waits on — sit behind
// q_ready/k_ready/p_done of tile t+1: the drain starts ~3.4k cyc late
// (pvtile_wait). With this ON, the seam body finishes tile t FIRST (PV(t,last)
// -> confirm -> pv_tile_done) and only then starts QK(t+1,i0). qk_done(gl)'s
// commit moves after the QK MMAs (the PV is already confirmed complete, so it
// tracks QK alone — slightly EARLIER fire than the fused EARLY_COMMIT); commit
// counts and per-barrier order are unchanged, so all phase algebra holds.
// Output note (markdown/21): the seam's timing occasionally realizes the
// codebase's pre-existing benign +-1ULP rounding wobble at kt64 (base realizes
// the same wobble at kt8), so the gate for this path is numerical equality
// (make validation + masked-ck / autopsy diff <=1ULP), not raw bit-identity.
#ifndef ATTENTION_FLAT_SEAM_PV_FIRST
#define ATTENTION_FLAT_SEAM_PV_FIRST 0
#endif
#if ATTENTION_FLAT_SEAM_PV_FIRST && !ATTENTION_CONTINUOUS_FLAT
#error "ATTENTION_FLAT_SEAM_PV_FIRST requires ATTENTION_CONTINUOUS_FLAT"
#endif
// SEAM_CONSUMER_REPLAY (markdown/21, adopted at 1071 TFLOPS): the seam QK
// commits to an isolated 128B-strided qk_seam_done[pipe] (init-once, nobody
// waits it while the drain runs); after the drain's closing bar.sync, ONE
// designated consumer warp per pipe waits it (in the common case QK completed
// mid-drain, so the wait returns instantly) and plain-arrives qk_done[pipe].
// qk_done's per-gl transition order and total count (G+1) are preserved: the
// seam gl's transition #gl+1 comes from the replay, and body gl+1's fused
// commit still follows it (its QK needs K(gl+1), whose TMA gate is transition
// #gl+1). At seams the K(gl+1)/Vh0(gl) TMA issue slides to post-drain and
// aligns with the O-store tail — measured faster than the direct-commit seam
// (1071 vs 1042).
#ifndef ATTENTION_FLAT_SEAM_CONSUMER_REPLAY
#define ATTENTION_FLAT_SEAM_CONSUMER_REPLAY 0
#endif
#if ATTENTION_FLAT_SEAM_CONSUMER_REPLAY && !ATTENTION_FLAT_SEAM_PV_FIRST
#error "ATTENTION_FLAT_SEAM_CONSUMER_REPLAY requires ATTENTION_FLAT_SEAM_PV_FIRST"
#endif
// (b) o_taddr early-free (markdown/21 §4-3): split the drain's completion
// signal so the producer's accumulate=false PV(t+1,i0) — which only needs the
// drain's tcgen05.lds of o_taddr to have RETIRED — no longer waits for the FP
// pack as well. The drain's last chunk is split into [ld+wait::ld -> arrive
// o_ld_done] then [scale/pack]; warp2's O store keeps waiting o_drained
// (pack+fence done). Expected ~neutral under SEAM_CONSUMER_REPLAY (there the
// binding chain for PV(t+1,i0) is [replay -> softmax(t+1,i0) -> p_done], and
// o_drained already fires before the replay) — kept as the correct signal
// split for any schedule where the o_taddr WAR is binding.
#ifndef ATTENTION_FLAT_O_LD_DONE
#define ATTENTION_FLAT_O_LD_DONE 0
#endif
#if ATTENTION_FLAT_O_LD_DONE && !ATTENTION_CONTINUOUS_FLAT
#error "ATTENTION_FLAT_O_LD_DONE requires ATTENTION_CONTINUOUS_FLAT"
#endif
// Manual per-tile timing probe for the flat path (CLOCK_TRACE is #error'd for FLAT).
// Records CTA0 pipe0's per-tile QK-start + drain start/end clocks -> printf the
// cadence and drain time at exit, to locate the flat's structural overhead vs base.
#ifndef ATTENTION_FLAT_PROBE
#define ATTENTION_FLAT_PROBE 0
#endif
#ifndef ATTENTION_FLAT_PROBE_GL
#define ATTENTION_FLAT_PROBE_GL 18   // target steady L2 iter (18%4==2) for within-iter probe
#endif
#if ATTENTION_CONTINUOUS_FLAT
#if !(ATTENTION_PERSISTENT && ATTENTION_PERSISTENT_OVERLAP &&                   \
      ATTENTION_PERSISTENT_OVERLAP_EARLY && ATTENTION_PERSISTENT_OVERLAP_O_IN_V)
#error "ATTENTION_CONTINUOUS_FLAT requires the live config: PERSISTENT + OVERLAP + EARLY + O_IN_V"
#endif
#if ATTENTION_CLOCK_TRACE || ATTENTION_PERSISTENT_OVERLAP_QK_PEEL ||            \
    ATTENTION_PEEL_SOFTMAX || (ATTENTION_CROSS_PIPE_PHASE != 0)
#error "ATTENTION_CONTINUOUS_FLAT does not support TRACE / PEEL / CROSS_PIPE_PHASE"
#endif
#endif

// Inter-Q-tile clock trace: capture TWO consecutive tiles of the SAME persistent
// CTA (blockIdx.x==0, tile_local_idx == ATTENTION_TRACE_TILE0 and +1) into two
// pages of the clock-trace buffer, sharing ONE clock base so the two tiles land
// on a common timeline (records store start-base) -> real cross-tile overlap.
#ifndef ATTENTION_CLOCK_TRACE_2TILE
#define ATTENTION_CLOCK_TRACE_2TILE 0
#endif
#ifndef ATTENTION_TRACE_TILE0
#define ATTENTION_TRACE_TILE0 6
#endif
#if ATTENTION_CLOCK_TRACE_2TILE && !ATTENTION_CLOCK_TRACE
#error "ATTENTION_CLOCK_TRACE_2TILE requires ATTENTION_CLOCK_TRACE"
#endif
#if ATTENTION_CLOCK_TRACE_2TILE && !ATTENTION_PERSISTENT_OVERLAP
#error "ATTENTION_CLOCK_TRACE_2TILE requires ATTENTION_PERSISTENT_OVERLAP"
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
#if ATTENTION_CONTINUOUS_FLAT && ATTENTION_FLAT_DEBUG
__device__ int g_dbg_gl[16];
#endif
#if ATTENTION_FLAT_PROBE
__device__ unsigned long long g_probe_qk[64];      // CTA0 pipe0: per-tile QK start
__device__ unsigned long long g_probe_dr0[64];     // CTA0: per-tile drain start
__device__ unsigned long long g_probe_dr1[64];     // CTA0: per-tile drain end
__device__ unsigned long long g_probe_gl[80];      // CTA0 pipe0: per-gl (iter) start
__device__ unsigned long long g_probe_seg[8];      // within-iter wait segments @gl==PROBE_GL
__device__ unsigned long long g_probe_sm0[64];     // CTA0: per-tile consumer softmax start
__device__ unsigned long long g_probe_dra[64];     // drain: after bar.sync1
__device__ unsigned long long g_probe_drb[64];     // drain: after pv_tile_done wait
__device__ unsigned long long g_probe_drc[64];     // drain: after pack (before fence/bar2)
__device__ unsigned long long g_probe_smqk;        // CTA0 consumer pipe0: accumulated qk_done wait
__device__ unsigned long long g_probe_smqk_n;      // ...count of waits (for average)
__device__ unsigned long long g_probe_tmaqk;       // CTA0 TMA pipe0: accumulated qk_done(K-buf) wait
__device__ unsigned long long g_probe_tmaqk_n;
__device__ unsigned long long g_probe_kready;      // CTA0 producer pipe0: accumulated k_ready wait
__device__ unsigned long long g_probe_kready_n;
// Barriered clock read: plain clock64() has no memory side-effect so the compiler
// reorders it freely (the per-gl vs seg deltas came out mathematically inconsistent
// => reordering). The "memory" clobber + volatile pin it to its source position.
__device__ __forceinline__ unsigned long long probe_clk() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}
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

// B3 (softmax-peel): in tile t's store-tail (after the drain-pack, while the O
// TMA store is async in flight) the consumer warps run tile (t+1)'s iter0/1
// softmax on the QK that was peeled during tile t's drain (commit ->
// qk_peel_done). This packs s_smem and reads p_taddr early, the precondition for
// B1's iter2/3 QK-peel. No new tcgen05.commit is issued by the consumer, so the
// deadlock risk is barrier phase accounting only. The p_done / s_h1_done arrives
// are NOT issued in the store-tail (they would be wiped by tile (t+1)'s
// body-start re-init); they are replayed at tile (t+1)'s body-start post-reinit
// (search "B3 body-start replay"), mirroring the qk_peel_done -> qk_done
// conversion. The peeled iter0/1 row_max/row_sum are carried into tile (t+1)'s
// consumer role via row_sum_partial (sum, shared) + the next tile's
// row_max_scratch (max, gmem -- no extra static smem), and that role then skips
// its own first iteration.
#ifndef ATTENTION_PEEL_SOFTMAX
#define ATTENTION_PEEL_SOFTMAX 0
#endif
#if ATTENTION_PEEL_SOFTMAX &&                                                  \
    !(ATTENTION_PERSISTENT_OVERLAP_QK_PEEL && ATTENTION_PEEL_DURING_DRAIN &&   \
      ATTENTION_PEEL_AFTER_SYNC && ATTENTION_PEEL_PREARM)
#error                                                                         \
    "ATTENTION_PEEL_SOFTMAX requires QK_PEEL + PEEL_DURING_DRAIN + PEEL_AFTER_SYNC + PEEL_PREARM"
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

// B3 mirrors the consumer first-iter COMPUTE_MAX + APPLY_SHIFT path verbatim, so
// it needs all three on (the FAST default). Checked here, after they are defined.
#if ATTENTION_PEEL_SOFTMAX &&                                                  \
    !(ATTENTION_FIRST_ITER_ROW_MAX_SHIFT &&                                    \
      ATTENTION_FIRST_ITER_APPLY_SHIFT && ATTENTION_FIRST_ITER_COMPUTE_MAX)
#error                                                                         \
    "ATTENTION_PEEL_SOFTMAX requires FIRST_ITER_ROW_MAX_SHIFT + APPLY_SHIFT + COMPUTE_MAX"
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
    unsigned int qk_done_carry,
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
#if ATTENTION_CLOCK_TRACE
  // Deferred-store completion: how long tile t blocks on tile (t-1)'s O store.
  // This is the residual (un-hidden) store tail; it lands at tile t's start, so on
  // the shared 2-tile timeline it shows the previous tile's store overlapping this one.
  const unsigned long long sw_start =
      (clock_trace != nullptr && pipe == 0 && lane0 && wait_prev_store)
          ? clock64()
          : 0ull;
#endif
  if (pipe == 0 && lane0 && wait_prev_store) {
    tma_store_wait_group_read();
  }
#if ATTENTION_CLOCK_TRACE
  if (clock_trace != nullptr && pipe == 0 && lane0 && wait_prev_store) {
    write_clock_trace_record(
        clock_trace, clock_trace_iters * kClockTraceSlotsPerIter + 11,
        kClockTraceStore, loop_repeats, pipe, role_warp_id, -1, -1, sw_start,
        clock64(), clock_trace_base);
  }
#endif
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
      mbarrier_wait(&qk_done[pipe], phase ^ qk_done_carry);
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
#if ATTENTION_FLAT_PROBE
    const bool _pk = blockIdx.x == 0 && pipe == 0 && lane == 0;
    const unsigned long long _kt0 = _pk ? probe_clk() : 0ull;
#endif
    mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_FLAT_PROBE
    if (_pk) { g_probe_kready += probe_clk() - _kt0; g_probe_kready_n += 1; }
#endif
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
    mbarrier_wait(&qk_done[pipe], tail_phase ^ q_ready_phase);
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
#if ATTENTION_PEEL_SOFTMAX
    bool softmax_peeled,
#endif
    unsigned int qk_done_carry,
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
#if ATTENTION_PEEL_SOFTMAX
  // B3: iter0/1 softmax (ld -> row_max -> pack+store -> p_done / s_h1_done) was
  // already run in the PREVIOUS tile's store-tail. Pick up its carried state and
  // resume at iter2 (local=1). The phantom qk_done arrive at the tile body start
  // keeps qk_done's phase clock aligned, so the phase math below is unchanged.
  if (softmax_peeled) {
    // Carry from the previous tile's store-tail peel: row_max via this tile's
    // own row_max_scratch (gmem, written there by the prior tile), row_sum via
    // row_sum_partial. row_max_scratch is read here before this role overwrites
    // it at the end, and the gmem load latency is hidden behind the qk_done wait
    // in the prefix/steady loop below.
    row_max_reg = row_max_scratch[pipe * kTileM + row];
    if (do_row_sum) row_sum_reg = row_sum_partial[pipe][row];
    iter = pipe + kActivePipeStride;
    local = 1;
  } else
#endif
  if (iter < loop_repeats) {
    mbarrier_wait(&qk_done[pipe], qk_done_carry);
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
      mbarrier_wait(&qk_done[pipe], phase ^ qk_done_carry);
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
#if ATTENTION_FLAT_PROBE
    const bool _pk = blockIdx.x == 0 && pipe == 0 && consumer_warp == 0 && lane == 0;
    const unsigned long long _t0 = _pk ? probe_clk() : 0ull;
#endif
    mbarrier_wait(&qk_done[pipe], phase ^ qk_done_carry);
#if ATTENTION_FLAT_PROBE
    if (_pk) { g_probe_smqk += probe_clk() - _t0; g_probe_smqk_n += 1; }
#endif
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
    mbarrier_wait(&qk_done[pipe], phase ^ qk_done_carry);
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
    unsigned int peel_q_phase, bool lane0
#if ATTENTION_CLOCK_TRACE_2TILE
    , ClockTraceRecord* peel_page = nullptr, int peel_slot_base = 0,
    unsigned long long peel_ct_base = 0
#endif
    ) {
#if ATTENTION_CLOCK_TRACE_2TILE
  // Split the during-drain peel into its real sub-phases: wait(Q TMA) / wait(K
  // TMA) / QK MMA issue. clock64 on lane0 brackets each (the waits are warp-wide
  // so lane0's stamp after a wait ~= when that operand became ready).
  const unsigned long long peel_t0 =
      (peel_page != nullptr && lane0) ? clock64() : 0ull;
#endif
  mbarrier_wait(q_ready, peel_q_phase);
#if ATTENTION_CLOCK_TRACE_2TILE
  unsigned long long peel_t1 = 0ull;
  if (peel_page != nullptr && lane0) {
    peel_t1 = clock64();
    write_clock_trace_record(peel_page, peel_slot_base + 0, kClockTracePeelQWait,
                             PIPE, PIPE, PIPE, -1, -1, peel_t0, peel_t1,
                             peel_ct_base);
  }
#endif
  mbarrier_wait(&k_ready[PIPE], 0u);
#if ATTENTION_CLOCK_TRACE_2TILE
  unsigned long long peel_t2 = 0ull;
  if (peel_page != nullptr && lane0) {
    peel_t2 = clock64();
    write_clock_trace_record(peel_page, peel_slot_base + 1, kClockTracePeelKWait,
                             PIPE, PIPE, PIPE, -1, -1, peel_t1, peel_t2,
                             peel_ct_base);
  }
#endif
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
#if ATTENTION_CLOCK_TRACE_2TILE
    if (peel_page != nullptr) {
      write_clock_trace_record(peel_page, peel_slot_base + 2, kClockTracePeelIssue,
                               PIPE, PIPE, PIPE, -1, -1, peel_t2, clock64(),
                               peel_ct_base);
    }
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

#if ATTENTION_CONTINUOUS_FLAT && ATTENTION_FLAT_DEBUG
__device__ __forceinline__ void flat_wait_dbg(uint64_t* bar, uint32_t phase,
                                              int line) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(bar);
  const unsigned long long start = clock64();
  for (;;) {
    unsigned done;
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(addr), "r"(phase)
        : "memory");
    if (done) break;
    if (clock64() - start > 1000000000ull) {
      if ((threadIdx.x & 31) == 0)
        printf("STUCK line=%d warp=%d gl=%d phase=%u\n", line,
               static_cast<int>(threadIdx.x >> 5),
               g_dbg_gl[threadIdx.x >> 5], phase);
      break;
    }
  }
#else
  (void)bar; (void)phase; (void)line;
#endif
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
#if ATTENTION_CONTINUOUS_FLAT
  // dep4 (o_taddr WAR): each consumer drain reads BOTH pipes' o_taddr, so all 8
  // consumer warps arrive this one barrier after the drain; tile (t+1)'s first PV
  // (accumulate=false, both pipes) and warp2's O store wait it. count = all
  // consumer warps. Advances once per tile.
  __shared__ uint64_t o_drained;
  // Per-tile PV-complete signal for the drain. pv_done cycles once per PV, so a
  // late drain (producer raced ahead) cannot track it with a level-triggered
  // wait. Instead each producer pipe, after confirming its tile's last PV
  // COMPLETED, arrives this once per tile; the drain waits it @(tile&1). count =
  // 2 (both producer pipes). Advances exactly once per tile.
  __shared__ uint64_t pv_tile_done;
#if ATTENTION_FLAT_SEAM_CONSUMER_REPLAY
  // Dedicated seam-QK commit target: 128B stride so the two pipes' barriers do
  // not share an adjacency window; nobody waits it until after the drain (the
  // post-drain consumer replay consumes it). See macro note.
  __shared__ __align__(128) uint64_t qk_seam_done[kPipeCount * 16];
#endif
#if ATTENTION_FLAT_O_LD_DONE
  // Drain-signal split: all 8 consumer warps' o_taddr tcgen05.lds RETIRED
  // (o_taddr free for PV(t+1,i0)'s accumulate=false overwrite); the pack may
  // still be running. warp2's O store keeps waiting o_drained (pack done).
  // count = all consumer warps; advances once per tile.
  __shared__ uint64_t o_ld_done;
#endif
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
  // Timestamp right after the async O store is ISSUED (commit_group returns).
  // Used as the O-store box end so it reflects the fire-and-forget issue and is
  // independent of the store-tail softmax peel that follows it on the consumer
  // warps (the real DMA is async / DEFER-waited in the next tile).
  __shared__ unsigned long long tma_store_issued_shared;
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
#if ATTENTION_CONTINUOUS_FLAT
  {
#if ATTENTION_FLAT_DEBUG
#define mbarrier_wait(b, p) flat_wait_dbg((b), (p), __LINE__)
#endif
    // ================= Continuous flatten pipeline (15_FLATTEN_IMPL.md) =======
    // Each role owns its own cross-tile loop; no outer per-tile loop, no CTA
    // boundary __syncthreads. QK(t+1) overlaps drain(t). Live config only.
    uintptr_t smem_addr =
        (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
    asm volatile("" : "+l"(smem_addr));
    uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
    uint32_t* k_smem[kPipeCount];
    uint32_t* v_smem[kPipeCount];
    uint32_t* s_smem[kPipeCount];
#pragma unroll
    for (int p = 0; p < kPipeCount; ++p) {
      k_smem[p] = q_smem + (1 + p) * kTileWords;
      v_smem[p] = q_smem + (1 + kKBufferTileCount + p) * kTileWords;
      s_smem[p] = q_smem + (1 + kKBufferTileCount + kVBufferCount + p) * kTileWords;
    }
    const uint32_t tmem_base = tmem_base_shared;
    const uint32_t p_taddr[kPipeCount] = {tmem_base, tmem_base + 128u};
    const uint32_t o_taddr[kPipeCount] = {tmem_base + 256u, tmem_base + 384u};

    if (threadIdx.x == 0) {
      mbarrier_init(&q_ready, 1);
      mbarrier_init(&qk_all_done, 2);
      mbarrier_init(&o_drained, kPipeCount * kConsumerWarpsPerPipe);
#if ATTENTION_FLAT_O_LD_DONE
      mbarrier_init(&o_ld_done, kPipeCount * kConsumerWarpsPerPipe);
#endif
      mbarrier_init(&pv_tile_done, kPipeCount);
#pragma unroll
      for (int p = 0; p < kPipeCount; ++p) {
        mbarrier_init(&k_ready[p], 1);
        mbarrier_init(&qk_done[p], 1);
        mbarrier_init(&p_done[p], kConsumerWarpsPerPipe);
        mbarrier_init(&s_h1_done[p], kConsumerWarpsPerPipe);
        mbarrier_init(&v_ready[p], 1);
        mbarrier_init(&v_h1_ready[p], 1);
        mbarrier_init(&pv_done[p], 1);
#if ATTENTION_FLAT_SEAM_CONSUMER_REPLAY
        mbarrier_init(&qk_seam_done[p * 16], 1);
#endif
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const int R = loop_repeats;
    const int bx = static_cast<int>(blockIdx.x);
    const int gx = static_cast<int>(gridDim.x);
    const int J = total_tiles > bx ? (total_tiles - bx + gx - 1) / gx : 0;
    const bool has_output = output != nullptr;
    float* const out_f32 =
        has_output ? reinterpret_cast<float*>(output) : nullptr;

    if (J > 0 && (warp_id == 0 || warp_id == 1)) {
      // =================== PRODUCER (fused flat, pipe = warp_id) =============
      const int pipe = warp_id;
      const uint32_t idesc = make_qk_idesc();
      const uint32_t pv_idesc = make_qk_idesc() | (1u << 16);
      const QkDescGen q_desc{static_cast<uint32_t>(smem_ptr_u32(q_smem) >> 4)};
      const QkDescGen k_desc{static_cast<uint32_t>(smem_ptr_u32(k_smem[pipe]) >> 4)};
      const PvSDescGen pv_s_desc{static_cast<uint32_t>(smem_ptr_u32(s_smem[pipe]) >> 4)};
      const PvVDescGen pv_v_desc{static_cast<uint32_t>(smem_ptr_u32(v_smem[pipe]) >> 4)};
      const int n_p = (R - pipe + kActivePipeStride - 1) / kActivePipeStride;
      const int G = J * n_p;
      unsigned int o_wait = 0u;  // running tile counter for o_drained waits
      // prologue gl=0: QK only (p_taddr fresh; q_ready / k_ready phase 0)
      mbarrier_wait(&q_ready, 0u);
      mbarrier_wait(&k_ready[pipe], 0u);
      if (lane0) {
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma)
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc, mma != 0);
        tcgen05_commit(&qk_done[pipe]);
      }
      // steady gl=1..G-1: QK(gl) + lagged PV(gl-1); tail PV after loop.
      for (int gl = 1; gl < G; ++gl) {
#if ATTENTION_FLAT_DEBUG
        g_dbg_gl[warp_id] = gl;
#endif
#if ATTENTION_FLAT_PROBE
        if (bx == 0 && pipe == 0 && lane0 && gl < 80) g_probe_gl[gl] = probe_clk();
#endif
        const int L = gl % n_p;
        const uint32_t ph = static_cast<uint32_t>(gl & 1);
        const uint32_t pph = static_cast<uint32_t>((gl - 1) & 1);
#if ATTENTION_FLAT_SEAM_PV_FIRST
        if (L == 0) {
          // Seam body (see ATTENTION_FLAT_SEAM_PV_FIRST note): tile j-1 first.
          const int j = gl / n_p;
          mbarrier_wait(&qk_done[pipe], pph);  // QK(gl-1) done -> q_smem free
          if (lane0) mbarrier_arrive(&qk_all_done);
          mbarrier_wait(&v_ready[pipe], pph);  // V(gl-1) h0
          if (lane0) {
#pragma unroll
            for (int mma = 0; mma < kMmasPerTile / 2; ++mma)
              tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                                  pv_idesc, true);  // tile-last PV: accumulate
          }
          mbarrier_wait(&s_h1_done[pipe], pph);
          mbarrier_wait(&v_h1_ready[pipe], pph);  // V(gl-1) h1
          if (lane0) {
#pragma unroll
            for (int mma = kMmasPerTile / 2; mma < kMmasPerTile; ++mma)
              tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                                  pv_idesc, true);
            tcgen05_commit(&pv_done[pipe]);
          }
          mbarrier_wait(&pv_done[pipe], pph);       // PV(gl-1) COMPLETED
          if (lane0) mbarrier_arrive(&pv_tile_done);  // drain(j-1) unblocked NOW
          // ...then tile j's first QK.
          mbarrier_wait(&q_ready, static_cast<uint32_t>(j & 1));
          mbarrier_wait(&k_ready[pipe], ph);
          mbarrier_wait(&p_done[pipe], pph);        // dep2: p_taddr WAR
          if (lane0) {
#if ATTENTION_FLAT_PROBE
            if (bx == 0 && pipe == 0 && (gl / n_p) < 64)
              g_probe_qk[gl / n_p] = probe_clk();   // per-tile QK start (cadence)
#endif
#pragma unroll
            for (int mma = 0; mma < kMmasPerTile; ++mma)
              tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc,
                                  mma != 0);
#if ATTENTION_FLAT_SEAM_CONSUMER_REPLAY
            // Isolated commit target; the post-drain consumer replay provides
            // the qk_done arrive. Without a drain (no output) there is no
            // replayer, so commit qk_done directly (uniform across the CTA).
            if (has_output) tcgen05_commit(&qk_seam_done[pipe * 16]);
            else tcgen05_commit(&qk_done[pipe]);
#else
            tcgen05_commit(&qk_done[pipe]);  // moved: covers QK(gl) alone
#endif
          }
          continue;
        }
#endif
        if (L == 0) {
          const int j = gl / n_p;
          // confirm prev tile's last QK completed (q_smem free), then signal it.
          mbarrier_wait(&qk_done[pipe], pph);
          if (lane0) mbarrier_arrive(&qk_all_done);
          mbarrier_wait(&q_ready, static_cast<uint32_t>(j & 1));
        }
        mbarrier_wait(&k_ready[pipe], ph);
#if ATTENTION_FLAT_PROBE
        if (gl == ATTENTION_FLAT_PROBE_GL && bx == 0 && pipe == 0 && lane0) g_probe_seg[0] = probe_clk();
#endif
        mbarrier_wait(&p_done[pipe], pph);       // dep2: p_taddr WAR
#if ATTENTION_FLAT_PROBE
        if (gl == ATTENTION_FLAT_PROBE_GL && bx == 0 && pipe == 0 && lane0) g_probe_seg[1] = probe_clk();
#endif
        if (lane0) {
#if ATTENTION_FLAT_PROBE
          if (L == 0 && bx == 0 && pipe == 0 && (gl / n_p) < 64)
            g_probe_qk[gl / n_p] = probe_clk();    // per-tile QK start (cadence)
#endif
#pragma unroll
          for (int mma = 0; mma < kMmasPerTile; ++mma)
            tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc, mma != 0);
        }
        // PV for gl-1 (lag); accumulate=false iff it is its tile's first iter.
        const int prevL = (gl - 1) % n_p;
        const int prevj = (gl - 1) / n_p;
        const bool pv_accum = (prevL != 0);
        mbarrier_wait(&v_ready[pipe], pph);       // V h0 ready (~free under PUSH)
#if ATTENTION_FLAT_PROBE
        if (gl == ATTENTION_FLAT_PROBE_GL && bx == 0 && pipe == 0 && lane0) g_probe_seg[2] = probe_clk();
#endif
        if (lane0) {
          if (!pv_accum && prevj > 0 && has_output) {
            // dep4: o_taddr WAR vs the previous drain's reads. With O_LD_DONE
            // the gate is "drain lds retired" (pack may still run).
#if ATTENTION_FLAT_O_LD_DONE
            mbarrier_wait(&o_ld_done, o_wait & 1u);
#else
            mbarrier_wait(&o_drained, o_wait & 1u);
#endif
            ++o_wait;
          }
#pragma unroll
          for (int mma = 0; mma < ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8; ++mma)
            tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                                pv_idesc, pv_accum || mma != 0);
          tcgen05_commit(&qk_done[pipe]);
#pragma unroll
          for (int mma = ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8; mma < kMmasPerTile / 2; ++mma)
            tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma], pv_idesc, true);
        }
        mbarrier_wait(&s_h1_done[pipe], pph);
#if ATTENTION_FLAT_PROBE
        if (gl == ATTENTION_FLAT_PROBE_GL && bx == 0 && pipe == 0 && lane0) g_probe_seg[3] = probe_clk();
#endif
        mbarrier_wait(&v_h1_ready[pipe], pph);
#if ATTENTION_FLAT_PROBE
        if (gl == ATTENTION_FLAT_PROBE_GL && bx == 0 && pipe == 0 && lane0) g_probe_seg[4] = probe_clk();
#endif
        if (lane0) {
#pragma unroll
          for (int mma = kMmasPerTile / 2; mma < kMmasPerTile; ++mma)
            tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma], pv_idesc, true);
          tcgen05_commit(&pv_done[pipe]);
        }
#if ATTENTION_FLAT_PROBE
        if (gl == ATTENTION_FLAT_PROBE_GL && bx == 0 && pipe == 0 && lane0) g_probe_seg[5] = probe_clk();
#endif
        // PV(gl-1) was tile (j-1)'s last iter when L==0: confirm it COMPLETED
        // (pv_done@pph flips after the MMAs finish) then signal the drain.
        if (L == 0) {
          mbarrier_wait(&pv_done[pipe], pph);
          if (lane0) mbarrier_arrive(&pv_tile_done);
        }
#if ATTENTION_FLAT_PROBE
        if (gl == ATTENTION_FLAT_PROBE_GL && bx == 0 && pipe == 0 && lane0) g_probe_seg[6] = probe_clk();
#endif
      }
      // tail: PV for gl=G-1
      {
        const int prevL = (G - 1) % n_p;
        const int prevj = (G - 1) / n_p;
        const uint32_t pph = static_cast<uint32_t>((G - 1) & 1);
        const bool pv_accum = (prevL != 0);
        // last tile's qk_all_done (never waited, but keeps count symmetric)
        if (lane0) mbarrier_arrive(&qk_all_done);
        mbarrier_wait(&v_ready[pipe], pph);
        if (lane0) {
          if (!pv_accum && prevj > 0 && has_output) {
#if ATTENTION_FLAT_O_LD_DONE
            mbarrier_wait(&o_ld_done, o_wait & 1u);
#else
            mbarrier_wait(&o_drained, o_wait & 1u);
#endif
            ++o_wait;
          }
#pragma unroll
          for (int mma = 0; mma < ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8; ++mma)
            tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma],
                                pv_idesc, pv_accum || mma != 0);
          tcgen05_commit(&qk_done[pipe]);
#pragma unroll
          for (int mma = ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER - 8; mma < kMmasPerTile / 2; ++mma)
            tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma], pv_idesc, true);
        }
        mbarrier_wait(&s_h1_done[pipe], pph);
        mbarrier_wait(&v_h1_ready[pipe], pph);
        if (lane0) {
#pragma unroll
          for (int mma = kMmasPerTile / 2; mma < kMmasPerTile; ++mma)
            tcgen05_mma_bf16_ss(o_taddr[pipe], pv_s_desc[mma], pv_v_desc[mma], pv_idesc, true);
          tcgen05_commit(&pv_done[pipe]);
        }
        // last tile's last PV: confirm completion, signal the drain.
        mbarrier_wait(&pv_done[pipe], pph);
        if (lane0) mbarrier_arrive(&pv_tile_done);
      }
    } else if (J > 0 && (warp_id == 2 || warp_id == 3)) {
      // =================== TMA (K/V load; warp2 also Q-prefetch + O store) ====
      const int pipe = warp_id - 2;
      const int n_p = (R - pipe + kActivePipeStride - 1) / kActivePipeStride;
      const int G = J * n_p;
      unsigned int store_wait = 0u;   // running tile counter for o_drained (store)
      unsigned int qa_wait = 0u;      // running tile counter for qk_all_done (Q prefetch)
      // warp2 issues Q for tile 0 up front. NOTE: mbarrier_expect_tx uses
      // elect.sync with a full-warp mask, so ALL lanes must call it (only the
      // tma_load_2d is lane0-gated).
      if (warp_id == 2) {
        mbarrier_expect_tx(&q_ready, kTileBytes);
        if (lane0) {
          const uint32_t q_addr = smem_ptr_u32(q_smem);
          tma_load_2d(&q_map, q_addr, &q_ready, 0, bx * kTileM);
          tma_load_2d(&q_map, q_addr + kTileBytes / 2, &q_ready, 32, bx * kTileM);
        }
      }
      // prologue gl=0: K(0) + V(0) h0/h1.
      {
        const int iter0 = pipe;
        const int gkt = kv_tile_base_for_block<kFixedKTiles>(bx, loop_k_tiles) +
                        local_k_tile_for_iter<kFixedKTiles>(iter0, loop_k_tiles);
        issue_k_tma_tile(&k_map, k_smem[pipe], &k_ready[pipe], gkt, lane0);
        issue_v_tma_half_tile(&v_map, v_smem[pipe], &v_ready[pipe], gkt, 0, lane0);
        issue_v_tma_half_tile(&v_map, v_smem[pipe], &v_h1_ready[pipe], gkt, 1, lane0);
      }
#if ATTENTION_FLAT_TMA_PUSH
      // F1 push-style body (see macro note): K(gl+1)+Vh0(gl) at qk_done(gl),
      // Vh1(gl) at pv_done(gl-1). Continuous: has_next is global (gl+1<G), never
      // per-tile, so there is no boundary ramp; the only seam specials are Q
      // prefetch + O store (warp2, L==0), same slots as the pull loop.
      for (int gl = 0; gl < G; ++gl) {
#if ATTENTION_FLAT_DEBUG
        g_dbg_gl[warp_id] = gl;
#endif
        const int L = gl % n_p;
        const int j = gl / n_p;
        const uint32_t ph = static_cast<uint32_t>(gl & 1);
        const uint32_t pph = static_cast<uint32_t>((gl - 1) & 1);
        const bool has_next = gl + 1 < G;
        // tile boundary: warp2 prefetches THIS tile's Q (after prev tile's last
        // QK). Q(0) already went out before the prologue.
        if (L == 0 && j > 0 && warp_id == 2) {
          mbarrier_wait(&qk_all_done, qa_wait & 1u);
          ++qa_wait;
          mbarrier_expect_tx(&q_ready, kTileBytes);  // all lanes (elect.sync)
          if (lane0) {
            const uint32_t q_addr = smem_ptr_u32(q_smem);
            const int q_row = (bx + j * gx) * kTileM;
            tma_load_2d(&q_map, q_addr, &q_ready, 0, q_row);
            tma_load_2d(&q_map, q_addr + kTileBytes / 2, &q_ready, 32, q_row);
          }
        }
        if (has_next) {
          // K(gl+1) (the next tile's iter0 when L==n_p-1) as soon as QK(gl)
          // completed: k_smem free (qk_done EARLY-commits mid PV(gl-1) h0).
          const int ntile = bx + ((gl + 1) / n_p) * gx;
          const int niter = ((gl + 1) % n_p) * kActivePipeStride + pipe;
          const int ngkt =
              kv_tile_base_for_block<kFixedKTiles>(ntile, loop_k_tiles) +
              local_k_tile_for_iter<kFixedKTiles>(niter, loop_k_tiles);
#if ATTENTION_FLAT_PROBE
          const bool _tk = bx == 0 && pipe == 0 && lane0;
          const unsigned long long _tt0 = _tk ? probe_clk() : 0ull;
#endif
          mbarrier_wait(&qk_done[pipe], ph);
#if ATTENTION_FLAT_PROBE
          if (_tk) { g_probe_tmaqk += probe_clk() - _tt0; g_probe_tmaqk_n += 1; }
#endif
          issue_k_tma_tile(&k_map, k_smem[pipe], &k_ready[pipe], ngkt, lane0);
        }
        // warp2 stores prev tile's O AFTER the K flow is unblocked (drain chain
        // needs the producer to reach PV(t,last): M1 bug#5) and BEFORE this
        // tile's pipe0 V overwrites v_smem[0] (store-WAR).
        if (L == 0 && j > 0 && warp_id == 2 && has_output && lane0) {
          mbarrier_wait(&o_drained, store_wait & 1u);  // tile (j-1) drain done
          ++store_wait;
          tma_store_4d(&o_map, smem_ptr_u32(v_smem[0]), 0, 0, bx + (j - 1) * gx, 0);
          tma_store_commit_group();
          tma_store_wait_group_read();  // store-WAR: done before tile j V(pipe0)
        }
        if (gl > 0) {  // V(0) went out in the prologue
          const int gkt =
              kv_tile_base_for_block<kFixedKTiles>(bx + j * gx, loop_k_tiles) +
              local_k_tile_for_iter<kFixedKTiles>(L * kActivePipeStride + pipe,
                                                  loop_k_tiles);
          if (has_next) {
            // EARLY Vh0(gl) (master 936): pre-pv_done. Overwrites v_smem h0
            // while PV(gl-1)'s tail h0 MMAs may still read it -- the exact race
            // master runs every steady iter (TMA flight > MMA tail, D1-proven).
            // The flat-new exposure is only the seam iter (L==0; pipe0's is
            // pushed past the race window by the store gate above anyway).
            issue_v_tma_half_tile(&v_map, v_smem[pipe], &v_ready[pipe], gkt, 0,
                                  lane0);
          }
          mbarrier_wait(&pv_done[pipe], pph);  // PV(gl-1) fully done
          if (!has_next)  // final iter's Vh0: post-pv_done (master 1033)
            issue_v_tma_half_tile(&v_map, v_smem[pipe], &v_ready[pipe], gkt, 0,
                                  lane0);
          issue_v_tma_half_tile(&v_map, v_smem[pipe], &v_h1_ready[pipe], gkt, 1,
                                lane0);
        }
      }
#else  // !ATTENTION_FLAT_TMA_PUSH: pull-style (Jun-30 M1 verified; K/V late)
      for (int gl = 1; gl < G; ++gl) {
#if ATTENTION_FLAT_DEBUG
        g_dbg_gl[warp_id] = gl;
#endif
        const int L = gl % n_p;
        const int j = gl / n_p;
        const uint32_t pph = static_cast<uint32_t>((gl - 1) & 1);
        // tile boundary: warp2 prefetches THIS tile's Q (after prev tile's last QK).
        if (L == 0 && warp_id == 2) {
          mbarrier_wait(&qk_all_done, qa_wait & 1u);
          ++qa_wait;
          mbarrier_expect_tx(&q_ready, kTileBytes);  // all lanes (elect.sync)
          if (lane0) {
            const uint32_t q_addr = smem_ptr_u32(q_smem);
            const int q_row = (bx + j * gx) * kTileM;
            tma_load_2d(&q_map, q_addr, &q_ready, 0, q_row);
            tma_load_2d(&q_map, q_addr + kTileBytes / 2, &q_ready, 32, q_row);
          }
        }
        const int tile = bx + j * gx;
        const int iter = L * kActivePipeStride + pipe;
        const int gkt = kv_tile_base_for_block<kFixedKTiles>(tile, loop_k_tiles) +
                        local_k_tile_for_iter<kFixedKTiles>(iter, loop_k_tiles);
#if ATTENTION_FLAT_PROBE
        const bool _tk = bx == 0 && pipe == 0 && lane0;
        const unsigned long long _tt0 = _tk ? probe_clk() : 0ull;
#endif
        mbarrier_wait(&qk_done[pipe], pph);      // K buffer free; also frees V h0
                                                 // region (producer EARLY-commits
                                                 // qk_done inside PV h0).
#if ATTENTION_FLAT_PROBE
        if (_tk) { g_probe_tmaqk += probe_clk() - _tt0; g_probe_tmaqk_n += 1; }
#endif
        issue_k_tma_tile(&k_map, k_smem[pipe], &k_ready[pipe], gkt, lane0);
        // warp2 stores prev tile's O AFTER issuing K (so the producer is unblocked
        // and can reach the PV/pv_tile_done the drain needs -> o_drained -> this
        // store), but BEFORE this tile's pipe0 V overwrites v_smem[0] (store-WAR).
        if (L == 0 && warp_id == 2 && has_output && lane0) {
          mbarrier_wait(&o_drained, store_wait & 1u);  // tile (j-1) drain done
          ++store_wait;
          tma_store_4d(&o_map, smem_ptr_u32(v_smem[0]), 0, 0, bx + (j - 1) * gx, 0);
          tma_store_commit_group();
          tma_store_wait_group_read();  // store-WAR: done before tile j V(pipe0)
        }
        // SAFE (Jun-30 M1 = 094ac4da): V h0 AND h1 issued AFTER pv_done, i.e. after
        // PV(gl-1) fully finished reading v_smem -> no WAR. V is late (the pull
        // -12%); TMA_PUSH is the perf path.
        mbarrier_wait(&pv_done[pipe], pph);      // PV(gl-1) done -> v_smem free
        issue_v_tma_half_tile(&v_map, v_smem[pipe], &v_ready[pipe], gkt, 0, lane0);
        issue_v_tma_half_tile(&v_map, v_smem[pipe], &v_h1_ready[pipe], gkt, 1, lane0);
      }
#endif  // ATTENTION_FLAT_TMA_PUSH
      // final tile (J-1) store (warp2): after its drain.
      if (warp_id == 2 && has_output) {
        mbarrier_wait(&o_drained, store_wait & 1u);
        if (lane0) {
          tma_store_4d(&o_map, smem_ptr_u32(v_smem[0]), 0, 0, bx + (J - 1) * gx, 0);
          tma_store_commit_group();
          tma_store_wait_group_read();
        }
      }
    } else if (J > 0 && warp_id >= kConsumerBaseWarp &&
               warp_id < kConsumerBaseWarp + kPipeCount * kConsumerWarpsPerPipe) {
      // =================== CONSUMER (reuse softmax role + drain) =============
      const int pipe = (warp_id - kConsumerBaseWarp) / kConsumerWarpsPerPipe;
      const int consumer_warp = (warp_id - kConsumerBaseWarp) - pipe * kConsumerWarpsPerPipe;
      const int n_p0 = (R + 1) / 2;
      const int n_p1 = R / 2;
      unsigned int pvph0 = 0u, pvph1 = 0u;  // running pv_done parity (drain)
      for (int j = 0; j < J; ++j) {
#if ATTENTION_FLAT_DEBUG
        g_dbg_gl[warp_id] = j;
#endif
        const int tile = bx + j * gx;
#if ATTENTION_FLAT_PROBE
        if (bx == 0 && warp_id == kConsumerBaseWarp && lane0 && j < 64)
          g_probe_sm0[j] = probe_clk();   // consumer softmax start (per tile)
#endif
        float* row_max_scratch =
            out_f32 != nullptr ? out_f32 + static_cast<size_t>(tile) * kTileWords : nullptr;
        attention_consumer_pipe_role(
            s_smem, qk_done, p_done, s_h1_done, row_sum_partial, row_max_scratch,
            p_taddr, o_taddr, pipe, consumer_warp, R, score_to_exp2_scale,
            has_output, nullptr, 0, 0, 0ull, 0u, lane);
        if (!has_output) continue;
        // ---- drain tile j (both pipes; all 8 consumer warps) ----
#if ATTENTION_FLAT_PROBE
        if (bx == 0 && warp_id == kConsumerBaseWarp && lane0 && j < 64)
          g_probe_dr0[j] = probe_clk();   // drain start
#endif
        asm volatile("bar.sync 1, 256;" ::: "memory");  // softmax partials visible
#if ATTENTION_FLAT_PROBE
        if (bx == 0 && warp_id == kConsumerBaseWarp && lane0 && j < 64) g_probe_dra[j] = probe_clk();
#endif
        // tile j's PVs (both pipes) completed -> producers arrived pv_tile_done
        // once for this tile (advances once/tile, so @(j&1) is unambiguous even if
        // a producer raced ahead).
        mbarrier_wait(&pv_tile_done, static_cast<uint32_t>(j & 1));
#if ATTENTION_FLAT_PROBE
        if (bx == 0 && warp_id == kConsumerBaseWarp && lane0 && j < 64) g_probe_drb[j] = probe_clk();
#endif
        uint32_t* output_bf16_smem = v_smem[0];
        const int epilogue_slot = warp_id - kConsumerBaseWarp;
        const int drain_warp = epilogue_slot & (kConsumerWarpsPerPipe - 1);
        const int drain_half = epilogue_slot / kConsumerWarpsPerPipe;
        const int row = drain_warp * 32 + lane;
        const float row_max0 = row_max_scratch[row];
        const float row_max1 = row_max_scratch[kTileM + row];
        const float common_row_max = fmaxf(row_max0, row_max1);
        const float pipe0_scale = exp2_approx_float_cpp(row_max0 - common_row_max);
        const float pipe1_scale = exp2_approx_float_cpp(row_max1 - common_row_max);
        const float denom = row_sum_partial[0][row] * pipe0_scale +
                            row_sum_partial[1][row] * pipe1_scale;
        const float inv_sum = denom != 0.0f ? 1.0f / denom : 0.0f;
        const uint32_t row_taddr0 = o_taddr[0] +
            (static_cast<uint32_t>(drain_warp * 32) << 16) +
            static_cast<uint32_t>(drain_half * 64);
        const uint32_t row_taddr1 = o_taddr[1] +
            (static_cast<uint32_t>(drain_warp * 32) << 16) +
            static_cast<uint32_t>(drain_half * 64);
        uint32_t* row_dst = output_bf16_smem +
            static_cast<size_t>(row) * (kTileN / 2) + drain_half * (kTileN / 4);
#if ATTENTION_FLAT_O_LD_DONE
#pragma unroll
        for (int chunk = 0; chunk < 3; ++chunk) {
          const uint32_t chunk_offset = static_cast<uint32_t>(chunk * 16);
          store_tmem_x16_pair_scale_norm_bf16_smem(
              row_taddr0 + chunk_offset, row_taddr1 + chunk_offset,
              row_dst + chunk * 8, pipe0_scale, pipe1_scale, inv_sum);
        }
        {
          // Last chunk split: retire the final o_taddr lds, signal o_ld_done
          // (o_taddr free for PV(t+1,i0)'s accumulate=false reset), then finish
          // the FP pack. Same math as the fused wrapper.
          uint32_t r0[16];
          uint32_t r1[16];
          TCGEN05_LD_X16(row_taddr0 + 48u, r0);
          TCGEN05_LD_X16(row_taddr1 + 48u, r1);
          tcgen05_wait_ld();
          if (lane0) mbarrier_arrive(&o_ld_done);
          uint32_t* dst = row_dst + 3 * 8;
#pragma unroll
          for (int i = 0; i < 16; i += 2) {
            const float lo = (__uint_as_float(r0[i]) * pipe0_scale +
                              __uint_as_float(r1[i]) * pipe1_scale) *
                             inv_sum;
            const float hi = (__uint_as_float(r0[i + 1]) * pipe0_scale +
                              __uint_as_float(r1[i + 1]) * pipe1_scale) *
                             inv_sum;
            dst[i >> 1] = pack_bf16_pair_device(lo, hi);
          }
        }
#else
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
          const uint32_t chunk_offset = static_cast<uint32_t>(chunk * 16);
          store_tmem_x16_pair_scale_norm_bf16_smem(
              row_taddr0 + chunk_offset, row_taddr1 + chunk_offset,
              row_dst + chunk * 8, pipe0_scale, pipe1_scale, inv_sum);
        }
#endif
#if ATTENTION_FLAT_PROBE
        if (bx == 0 && warp_id == kConsumerBaseWarp && lane0 && j < 64) g_probe_drc[j] = probe_clk();
#endif
        tma_store_fence();
        if (lane0) mbarrier_arrive(&o_drained);
        // ensure all drains done before next tile's softmax reuses o_taddr / partials
        asm volatile("bar.sync 1, 256;" ::: "memory");
#if ATTENTION_FLAT_SEAM_CONSUMER_REPLAY
        // Post-drain relay of the NEXT tile's seam-QK completion onto qk_done
        // (PPAS conditions: the commit landed mid-drain with no waiter; in the
        // common case QK finished during the drain so this returns instantly).
        // One designated warp per pipe; the other warps enter softmax(j+1) and
        // block on qk_done until this arrive. Seam commit #(j+1) -> parity j&1.
        if (j + 1 < J && consumer_warp == 0) {
          mbarrier_wait(&qk_seam_done[pipe * 16], static_cast<uint32_t>(j & 1));
          if (lane0) mbarrier_arrive(&qk_done[pipe]);
        }
#endif
#if ATTENTION_FLAT_PROBE
        if (bx == 0 && warp_id == kConsumerBaseWarp && lane0 && j < 64)
          g_probe_dr1[j] = probe_clk();   // drain end
#endif
      }
    }
#if ATTENTION_FLAT_DEBUG
#undef mbarrier_wait
#endif
    // Free tmem (D1 does this after the per-tile loop; the flat path returns
    // before reaching it, which would leave tmem allocated -> "tensor memory not
    // completely freed"). All 384 threads converge here first.
    if (threadIdx.x == 0) tcgen05_fence_after_thread_sync();
    __syncthreads();
#if ATTENTION_FLAT_PROBE
    if (bx == 0 && threadIdx.x == 0) {
      // producer(warp0) & consumer(warp4) share CTA0's clock -> aligned timeline.
      for (int t = 2; t < J && t < 8; ++t)
        printf("PROBE t%d cad=%llu sm=%llu | drain bar1=%llu pvtile=%llu pack=%llu "
               "fence+bar2=%llu tot=%llu\n",
               t, g_probe_qk[t] - g_probe_qk[t - 1],
               g_probe_dr0[t] - g_probe_sm0[t], g_probe_dra[t] - g_probe_dr0[t],
               g_probe_drb[t] - g_probe_dra[t], g_probe_drc[t] - g_probe_drb[t],
               g_probe_dr1[t] - g_probe_drc[t], g_probe_dr1[t] - g_probe_dr0[t]);
      printf("PROBE softmax qk_wait avg=%llu (n=%llu) | TMA Kbuf_wait avg=%llu (n=%llu)\n",
             g_probe_smqk_n ? g_probe_smqk / g_probe_smqk_n : 0ull, g_probe_smqk_n,
             g_probe_tmaqk_n ? g_probe_tmaqk / g_probe_tmaqk_n : 0ull, g_probe_tmaqk_n);
      const int n_pp = (R + 1) / 2;  // pipe0 n_p
      for (int g = 14; g < 26; ++g)
        printf("PROBE gl%d(L%d) dur=%llu\n", g, g % n_pp,
               g_probe_gl[g] - g_probe_gl[g - 1]);
      printf("PROBE seg@gl%d k_ready=%llu p_done=%llu QK+vready=%llu PVh0+sh1=%llu "
             "vh1ready=%llu PVh1=%llu body=%llu\n",
             ATTENTION_FLAT_PROBE_GL,
             g_probe_seg[0] - g_probe_gl[ATTENTION_FLAT_PROBE_GL],
             g_probe_seg[1] - g_probe_seg[0], g_probe_seg[2] - g_probe_seg[1],
             g_probe_seg[3] - g_probe_seg[2], g_probe_seg[4] - g_probe_seg[3],
             g_probe_seg[5] - g_probe_seg[4],
             g_probe_seg[6] - g_probe_gl[ATTENTION_FLAT_PROBE_GL]);
    }
    __syncthreads();
#endif
    if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
    __syncthreads();
    if (warp_id == 0) tcgen05_relinquish_alloc_permit();
    return;
  }
#endif
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
  // Continuous inter-q-tile pipeline (14_CONTINUOUS_DESIGN.md): ALL pipeline
  // barriers are init-once (first tile only) and cycle continuously across tile
  // boundaries -- no per-tile re-init. The phase is carried by register
  // (phase_carry below) for the odd-per-tile barriers, exactly as q_ready
  // already does via q_ready_phase. This replaces the old per-tile re-init +
  // peel-bridge scaffolding (qk_peel_done / AFTER_SYNC peel / store-tail
  // softmax / body-start conversion+replay), which existed only because the old
  // boundary wiped these barriers.
  if (tile == static_cast<int>(blockIdx.x) && threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
#if ATTENTION_PERSISTENT_OVERLAP_EARLY
    mbarrier_init(&qk_all_done, 2);
#endif
#pragma unroll
    for (int p = 0; p < kPipeCount; ++p) {
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&qk_done[p], 1);
      mbarrier_init(&p_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&s_h1_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&v_ready[p], 1);
      mbarrier_init(&v_h1_ready[p], 1);
      mbarrier_init(&pv_done[p], 1);
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
#if ATTENTION_PERSISTENT_OVERLAP_EARLY && !ATTENTION_PERSISTENT_OVERLAP
    mbarrier_init(&qk_all_done, 2);
#endif
#if ATTENTION_PIPE1_TMA_HEAD_MARKER
    mbarrier_init(&tma_head_marker, 1);
#endif
#pragma unroll
    for (int p = 0; p < kPipeCount; ++p) {
#if !ATTENTION_PERSISTENT_OVERLAP
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&qk_done[p], 1);
      mbarrier_init(&p_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&s_h1_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&v_ready[p], 1);
      mbarrier_init(&v_h1_ready[p], 1);
      mbarrier_init(&pv_done[p], 1);
#endif
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
#if ATTENTION_PEEL_SOFTMAX
  // B3 body-start replay. Tile t's store-tail already ran THIS tile's iter0/1
  // softmax (ld p_taddr -> pack s_smem), but deferred the p_done / s_h1_done
  // arrives because those barriers are re-initialized just above (every tile,
  // thread 0), which would have wiped a store-tail arrive. Replay them here,
  // post-reinit, so the producer's first steady iter (waits p_done[pipe] and
  // s_h1_done[pipe] phase 0) is released. The consumer (softmax_peeled) skips
  // iter0/1 and never arrives them in its body, so this replay is the only
  // source. Each pipe's kConsumerWarpsPerPipe consumer warps arrive once (lane0)
  // -> the barrier count (kConsumerWarpsPerPipe) is met -> phase 0 completes,
  // exactly as the in-body iter0/1 softmax would have. s_smem (packed in the
  // store-tail) is already visible after the __syncthreads above. This mirrors
  // the qk_peel_done -> qk_done conversion right before it.
  if (qk_peeled && warp_id >= kConsumerBaseWarp &&
      warp_id < kConsumerBaseWarp + kPipeCount * kConsumerWarpsPerPipe) {
    const int sm_pipe = (warp_id - kConsumerBaseWarp) / kConsumerWarpsPerPipe;
    if (lane0) {
      mbarrier_arrive(&p_done[sm_pipe]);
      mbarrier_arrive(&s_h1_done[sm_pipe]);
    }
  }
#endif
#if ATTENTION_CLOCK_TRACE
#if ATTENTION_CLOCK_TRACE_2TILE
  const int ct_tl_idx =
      (tile - static_cast<int>(blockIdx.x)) / static_cast<int>(gridDim.x);
  const bool ct_is_tile0 =
      (static_cast<int>(blockIdx.x) == 0) && (ct_tl_idx == ATTENTION_TRACE_TILE0);
  const bool ct_is_tile1 =
      (static_cast<int>(blockIdx.x) == 0) && (ct_tl_idx == ATTENTION_TRACE_TILE0 + 1);
  if (threadIdx.x == 0) {
    if (ct_is_tile0) clock_trace_base_shared = clock64();
    if (ct_is_tile0 || ct_is_tile1) {
#pragma unroll
      for (int i = 0; i < kPipeCount * 2; ++i) k_tma_start_shared[i] = 0ull;
    }
  }
  __syncthreads();
#else
  if (threadIdx.x == 0) {
    clock_trace_base_shared = clock64();
#pragma unroll
    for (int i = 0; i < kPipeCount * 2; ++i) {
      k_tma_start_shared[i] = 0ull;
    }
  }
  __syncthreads();
#endif
#endif
  const unsigned long long clock_trace_base = clock_trace_base_shared;
#if ATTENTION_CLOCK_TRACE_2TILE
  ClockTraceRecord* const clock_trace_eff =
      ct_is_tile0
          ? clock_trace
          : (ct_is_tile1 && clock_trace != nullptr
                 ? clock_trace + (clock_trace_iters * kClockTraceSlotsPerIter +
                                  kClockTraceExtraSlots)
                 : nullptr);
#else
  ClockTraceRecord* const clock_trace_eff = clock_trace;
#endif
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
  // Continuous pipeline phase carry for odd-per-tile barriers (qk_done,
  // qk_all_done): == tile_local_idx&1, the offset vs the role's phase-0-start
  // assumption now that barriers are init-once (no per-tile re-init). The
  // even-per-tile barriers (p_done/s_h1_done/v_ready/v_h1_ready/pv_done/k_ready)
  // return to phase 0 each tile for power-of-2 R>=8 (n_p=R/2 even), so they need
  // no carry -- just the re-init removal. See 14_CONTINUOUS_DESIGN.md §E.
  const unsigned int phase_carry = q_ready_phase;

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
        output != nullptr, clock_trace_eff, clock_trace_iters, clock_trace_start,
        clock_trace_base,
#if ATTENTION_PEEL_SOFTMAX
        qk_peeled,
#endif
        phase_carry,
        lane);
  }

  if (warp_id == 2 || warp_id == 3) {
    const int pipe = warp_id - 2;
    attention_pv_pipe_role<kFixedKTiles>(
        &k_map, &v_map, k_smem, v_smem, k_ready, v_ready, v_h1_ready,
        qk_done, pv_done,
        k_issue_gen, v_issue_gen, &tma_head_marker,
        k_tma_start_shared, pipe, loop_repeats, loop_k_tiles, kv_tile_base, clock_trace_eff,
        clock_trace_iters, clock_trace_start, clock_trace_base, k_prefetched,
#if ATTENTION_PERSISTENT_OVERLAP_DEFER_STORE
        wait_prev_store,
#endif
#if ATTENTION_PEEL_HB
        qk_peeled,
#endif
        phase_carry,
        lane);
  }

	  if (warp_id == 0 || warp_id == 1) {
	    const int pipe = warp_id;
	    attention_qk_pipe_role<kFixedKTiles>(
	        q_smem, k_smem, &q_ready, k_ready, qk_done, p_done, s_h1_done, pv_done,
		        qk_issue_gen,
	        s_smem, v_smem, v_ready, v_h1_ready, p_taddr, o_taddr, pipe,
        loop_repeats, clock_trace_eff,
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
  const bool trace_cta = clock_trace_eff != nullptr;
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
        mbarrier_wait(&qk_all_done, phase_carry);
        if (warp_id == 2) {
          mbarrier_expect_tx(&q_ready, kTileBytes);
          if (lane0) {
            const uint32_t q_smem_addr = smem_ptr_u32(q_smem);
            const int next_q_row = next_tile * kTileM;
#if ATTENTION_CLOCK_TRACE_2TILE
            // tile (t+1)'s Q is TMA-prefetched here, during tile t's epilogue.
            // Record the issue onto tile (t+1)'s trace page so the SVG shows the
            // Q load launching inside tile t's drain window (the "fill hide").
            ClockTraceRecord* const q_pf_page =
                (clock_trace != nullptr && ct_is_tile0)
                    ? clock_trace + (clock_trace_iters * kClockTraceSlotsPerIter +
                                     kClockTraceExtraSlots)
                    : nullptr;
            const unsigned long long q_pf_start =
                (q_pf_page != nullptr) ? clock64() : 0ull;
#endif
            tma_load_2d(&q_map, q_smem_addr, &q_ready, 0, next_q_row);
            tma_load_2d(&q_map, q_smem_addr + kTileBytes / 2, &q_ready, 32, next_q_row);
#if ATTENTION_CLOCK_TRACE_2TILE
            if (q_pf_page != nullptr) {
              write_clock_trace_record(
                  q_pf_page, clock_trace_iters * kClockTraceSlotsPerIter + 12,
                  kClockTraceQTma, -1, -1, 2, -1, -1, q_pf_start, clock64(),
                  clock_trace_base);
            }
#endif
          }
        }
        const int pf_pipe = warp_id - 2;
        const int next_kv_base =
            kv_tile_base_for_block<kFixedKTiles>(next_tile, loop_k_tiles);
        const int pf_k_tile =
            local_k_tile_for_iter<kFixedKTiles>(pf_pipe, loop_k_tiles);
#if ATTENTION_CLOCK_TRACE_2TILE
        // EARLY first-K prefetch = the peel's K1 (warp2->pipe0) / K2 (warp3->
        // pipe1). Record the issue onto tile (t+1)'s page so the SVG shows the
        // peel's K TMA, separate from the QK MMA. slot = pf_pipe*64 + 5.
        ClockTraceRecord* const k_pf_page =
            (clock_trace != nullptr && ct_is_tile0)
                ? clock_trace + (clock_trace_iters * kClockTraceSlotsPerIter +
                                 kClockTraceExtraSlots)
                : nullptr;
        const unsigned long long k_pf_start =
            (k_pf_page != nullptr && lane0) ? clock64() : 0ull;
#endif
        issue_k_tma_tile(&k_map, k_smem[pf_pipe], &k_ready[pf_pipe],
                         next_kv_base + pf_k_tile, lane0);
#if ATTENTION_CLOCK_TRACE_2TILE
        if (k_pf_page != nullptr && lane0) {
          write_clock_trace_record(
              k_pf_page, clock_trace_iters * kClockTraceSlotsPerIter + 19 + pf_pipe,
              kClockTracePeelKTma, pf_pipe, pf_pipe, 2 + pf_pipe, -1, -1,
              k_pf_start, clock64(), clock_trace_base);
        }
#endif
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
              clock_trace_eff,
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
              clock_trace_eff,
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
      write_clock_trace_record(clock_trace_eff, trace_extra_base, kClockTraceTailWait,
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
#if ATTENTION_CLOCK_TRACE_2TILE
        // The iter0/1 QK of tile (t+1) is peeled here (issued by tile t's w0/w1
        // during tile t's drain). The helper records its sub-phases onto tile
        // (t+1)'s page: wait(Q TMA) / wait(K TMA) / QK MMA issue, at slots
        // warp_id*64 + {0,1,2}. Only when THIS tile is the trace's tile0.
        ClockTraceRecord* const peel_trace_page =
            (clock_trace != nullptr && ct_is_tile0)
                ? clock_trace + (clock_trace_iters * kClockTraceSlotsPerIter +
                                 kClockTraceExtraSlots)
                : nullptr;
        // Write into the page's EXTRA region (alongside q_tma at +12), NOT the
        // iter0/1 per-iter block — tile (t+1) still runs iter0/1's PV/done marks
        // there and would overwrite the peel records. warp0 -> +13..15, warp1 -> +16..18.
        const int peel_slot_base = clock_trace_iters * kClockTraceSlotsPerIter +
                                   13 + warp_id * 3;
#endif
        if (warp_id == 0) {
          attention_issue_qk_peel<0>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0
#if ATTENTION_CLOCK_TRACE_2TILE
                                     , peel_trace_page, peel_slot_base,
                                     clock_trace_base
#endif
          );
        } else {
          attention_issue_qk_peel<1>(q_smem, k_smem, p_taddr, &q_ready, k_ready,
                                     qk_peel_done, peel_q_phase, lane0
#if ATTENTION_CLOCK_TRACE_2TILE
                                     , peel_trace_page, peel_slot_base,
                                     clock_trace_base
#endif
          );
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
      write_clock_trace_record(clock_trace_eff, trace_extra_base + 1 + consumer_warp,
                               kClockTraceTmemDrain, loop_repeats, -1, warp_id,
                               consumer_warp, -1, epilogue_start, epilogue_end,
                               clock_trace_base);
      write_clock_trace_record(clock_trace_eff, trace_extra_base + 5 + consumer_warp,
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
#if ATTENTION_CLOCK_TRACE
      if (trace_cta) tma_store_issued_shared = clock64();
#endif
    }
#if ATTENTION_PEEL_SOFTMAX
    // B3 store-tail softmax peel. The O TMA store above is async (DEFER_STORE
    // also defers its wait to the next tile), so this window is tensor-core /
    // consumer idle. The 8 consumer warps run tile (t+1)'s iter0/1 softmax on
    // the QK peeled during this tile's drain (AFTER_SYNC peel -> qk_peel_done).
    // This mirrors the consumer first-iter block (attention_consumer_pipe_role,
    // FIRST_ITER_COMPUTE_MAX + APPLY_SHIFT path): two row_max loads, then two
    // pack+store+shift halves that arrive p_done (h1) and free p_taddr[pipe]
    // early. row_max/row_sum are carried forward; tile (t+1)'s consumer skips
    // its first iter. No new tcgen05.commit here -> phase-accounting risk only.
    if (epilogue_warp && loop_repeats >= kActivePipeStride) {
      const int peel_next_tile = tile + static_cast<int>(gridDim.x);
      if (peel_next_tile < total_tiles) {
        const int sm_pipe =
            (warp_id - kConsumerBaseWarp) / kConsumerWarpsPerPipe;
        const int sm_warp =
            (warp_id - kConsumerBaseWarp) - sm_pipe * kConsumerWarpsPerPipe;
        const int sm_row = sm_warp * 32 + lane;
#if ATTENTION_CLOCK_TRACE_2TILE
        const bool sm_trace = trace_cta && lane0;
#endif
        // Wait for the peeled QK MMA result in p_taddr[sm_pipe] (commit ->
        // qk_peel_done[sm_pipe], phase 0). The body-start conversion still
        // re-arms qk_peel_done after the next __syncthreads, so the phase is
        // consistent across tiles (see lines re: PREARM re-init).
        mbarrier_wait(&qk_peel_done[sm_pipe], 0u);
        const uint32_t sm_row_taddr =
            p_taddr[sm_pipe] + (static_cast<uint32_t>(sm_warp * 32) << 16);
        // row max over both halves (untraced, like the body's COMPUTE_MAX --
        // the two halves of one row share a single max, so both reads must
        // precede either pack).
        float sm_row_max = tcgen05_ld_x64_wait_row_max_scaled_nvcc(
            sm_row_taddr, score_to_exp2_scale);
        sm_row_max =
            fmaxf(sm_row_max, tcgen05_ld_x64_wait_row_max_scaled_nvcc(
                                  sm_row_taddr + 64u, score_to_exp2_scale));
        // Per-half ld -> exp2+pack, each traced separately so the SVG renders
        // "ld -> softmax -> ld -> softmax" (the body first-iter shape) instead of
        // one merged block. DO NOT arrive p_done / s_h1_done here: tile (t+1)'s
        // body-start re-init (every tile, thread 0) would wipe them and hang the
        // producer (waits p_done/s_h1_done phase 0). They are replayed post-reinit
        // at tile (t+1)'s body-start (search "B3 body-start replay"), mirroring
        // the qk_peel_done -> qk_done conversion. Data is safe: p_taddr fully read,
        // s_smem fully packed, and the body-start __syncthreads makes the s_smem
        // writes visible to the producer.
#if ATTENTION_CLOCK_TRACE_2TILE
        const int sm_detail_base =
            trace_extra_base + kClockTracePeelSoftmaxDetailBase +
            (warp_id - kConsumerBaseWarp) * 4;
#endif
        float sm_sum0;
        float sm_sum1;
        {
          uint32_t r[64];
#if ATTENTION_CLOCK_TRACE_2TILE
          const unsigned long long ld0_s = sm_trace ? clock64() : 0ull;
#endif
          TCGEN05_LD_X64(sm_row_taddr, r);
          tcgen05_wait_ld();
#if ATTENTION_CLOCK_TRACE_2TILE
          const unsigned long long ld0_e = sm_trace ? clock64() : 0ull;
#endif
          sm_sum0 = pack_store_x64_loop_shifted<true>(
              s_smem[sm_pipe] + s_store_word_offset(sm_row, 0), r,
              score_to_exp2_scale, sm_row_max);
#if ATTENTION_CLOCK_TRACE_2TILE
          if (sm_trace) {
            write_clock_trace_record(clock_trace_eff, sm_detail_base + 0,
                                     kClockTracePeelSoftmaxLd, sm_pipe, sm_pipe,
                                     warp_id, sm_warp, 0, ld0_s, ld0_e,
                                     clock_trace_base);
            write_clock_trace_record(clock_trace_eff, sm_detail_base + 1,
                                     kClockTracePeelSoftmax, sm_pipe, sm_pipe,
                                     warp_id, sm_warp, 0, ld0_e, clock64(),
                                     clock_trace_base);
          }
          const unsigned long long ld1_s = sm_trace ? clock64() : 0ull;
#endif
          TCGEN05_LD_X64(sm_row_taddr + 64u, r);
          tcgen05_wait_ld();
#if ATTENTION_CLOCK_TRACE_2TILE
          const unsigned long long ld1_e = sm_trace ? clock64() : 0ull;
#endif
          sm_sum1 = pack_store_x64_loop_shifted<true>(
              s_smem[sm_pipe] + s_store_word_offset(sm_row, 32), r,
              score_to_exp2_scale, sm_row_max);
#if ATTENTION_CLOCK_TRACE_2TILE
          if (sm_trace) {
            write_clock_trace_record(clock_trace_eff, sm_detail_base + 2,
                                     kClockTracePeelSoftmaxLd, sm_pipe, sm_pipe,
                                     warp_id, sm_warp, 1, ld1_s, ld1_e,
                                     clock_trace_base);
            write_clock_trace_record(clock_trace_eff, sm_detail_base + 3,
                                     kClockTracePeelSoftmax, sm_pipe, sm_pipe,
                                     warp_id, sm_warp, 1, ld1_e, clock64(),
                                     clock_trace_base);
          }
#endif
        }
        // Carry to tile (t+1): row_max into ITS row_max_scratch (gmem, same
        // thread reads it back next iteration -> no fence needed), row_sum into
        // row_sum_partial (shared, free here since the drain-pack already read
        // tile t's). row_max_scratch = output base + tile*kTileWords, so the
        // next tile's scratch is +gridDim.x*kTileWords.
        float* const sm_next_scratch =
            row_max_scratch + static_cast<size_t>(gridDim.x) * kTileWords;
        sm_next_scratch[sm_pipe * kTileM + sm_row] = sm_row_max;
        row_sum_partial[sm_pipe][sm_row] = sm_sum0 + sm_sum1;
      }
    }
#endif
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
  }
  __syncthreads();
#if ATTENTION_FLAT_PROBE
  if (blockIdx.x == 0 && threadIdx.x == 0)
    printf("PROBEBASE k_ready avg=%llu (n=%llu) | softmax qk_wait avg=%llu (n=%llu)\n",
           g_probe_kready_n ? g_probe_kready / g_probe_kready_n : 0ull, g_probe_kready_n,
           g_probe_smqk_n ? g_probe_smqk / g_probe_smqk_n : 0ull, g_probe_smqk_n);
#endif

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
    // O-store box ends at the async ISSUE (commit_group), NOT at store_end:
    // store_end is sampled after the store-tail peel + __syncthreads, which would
    // make the box absorb the peel time even though the DMA is async/deferred.
    write_clock_trace_record(clock_trace_eff, trace_extra_base + 9,
                             kClockTraceGlobalStore, loop_repeats, -1, 0, -1, -1,
                             tma_store_start_shared, tma_store_issued_shared,
                             clock_trace_base);
    write_clock_trace_record(clock_trace_eff, trace_extra_base + 10,
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
