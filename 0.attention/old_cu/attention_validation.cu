#define main attention_custom_kernel_perf_main
#include "attention_custom_kernel.cu"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kValidationThreads = 128;
constexpr size_t kValidationSmemBytes = 2 * kTileBytes + 1024;
constexpr float kQValue = 0.0625f;
constexpr float kKValue = 0.0625f;
constexpr float kVValue = 0.125f;
constexpr float kQkAbsTol = 0.08f;
constexpr float kSoftmaxRelTol = 2.0e-2f;
constexpr float kPvRelTol = 3.0e-2f;
constexpr int kProbeQRow = 32;
constexpr int kProbeKCol = -1;

enum class ValidationPattern {
  Constant,
  Rank1,
  Onehot,
};

struct ValidationArgs {
  std::string stage = "all";
  ValidationPattern pattern = ValidationPattern::Constant;
  int k_tiles = 1;
  int probe_row = kProbeQRow;
  int probe_col = kProbeKCol;
  float softmax_m = 0.0f;
  bool row_max = false;
  std::string csv = "0.attention/attention_validation.csv";
  int dump_mismatch = 8;
};

int g_probe_q_row = kProbeQRow;
int g_probe_q_col = kProbeKCol;

struct CompareResult {
  std::string stage;
  bool ok = true;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  size_t bad_count = 0;
  int first_bad_row = -1;
  int first_bad_col = -1;
};

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--stage qk|softmax|pv|fused|all|probe]\n"
      "          [--pattern constant|rank1|onehot] [--k-tiles N] [--softmax-m M]\n"
      "          [--row-max] [--probe-row ROW] [--probe-col COL]\n"
      "          [--csv PATH] [--dump-mismatch N]\n",
      argv0);
}

ValidationPattern parse_pattern(const char* value) {
  const std::string pattern = value;
  if (pattern == "constant") return ValidationPattern::Constant;
  if (pattern == "rank1") return ValidationPattern::Rank1;
  if (pattern == "onehot") return ValidationPattern::Onehot;
  std::fprintf(stderr, "Invalid pattern: %s\n", value);
  std::exit(2);
}

ValidationArgs parse_validation_args(int argc, char** argv) {
  ValidationArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (opt == "--stage") {
      args.stage = need_value("--stage");
    } else if (opt == "--pattern") {
      args.pattern = parse_pattern(need_value("--pattern"));
    } else if (opt == "--k-tiles") {
      args.k_tiles = std::max(1, std::atoi(need_value("--k-tiles")));
    } else if (opt == "--probe-row") {
      args.probe_row = std::atoi(need_value("--probe-row"));
    } else if (opt == "--probe-col") {
      args.probe_col = std::atoi(need_value("--probe-col"));
    } else if (opt == "--softmax-m") {
      args.softmax_m = std::atof(need_value("--softmax-m"));
    } else if (opt == "--row-max") {
      args.row_max = true;
    } else if (opt == "--csv") {
      args.csv = need_value("--csv");
    } else if (opt == "--dump-mismatch") {
      args.dump_mismatch = std::max(0, std::atoi(need_value("--dump-mismatch")));
    } else if (opt == "--help" || opt == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", opt.c_str());
      usage(argv[0]);
      std::exit(2);
    }
  }
  if (args.stage != "qk" && args.stage != "softmax" && args.stage != "pv" &&
      args.stage != "fused" && args.stage != "all" && args.stage != "probe") {
    std::fprintf(stderr, "Invalid stage: %s\n", args.stage.c_str());
    std::exit(2);
  }
  return args;
}

uint16_t float_to_bf16_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_bits_to_float(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

uint32_t pack_bf16_pair(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) |
         (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}

__host__ __device__ __forceinline__ int logical_word_offset(int row, int col_pair) {
  // tcgen05 reads this tile as eight consecutive K16 atoms.  Each atom is
  // 128 rows x 16 bf16 = 1024 packed u32 words, and the MMA descriptors are
  // issued at base + atom * 1024 words.  Validation keeps the atom itself
  // unswizzled so layout debugging is about logical placement, not bank layout.
  const int k16_atom = col_pair >> 3;
  const int pair_in_atom = col_pair & 7;
  const int row_group8 = row >> 3;
  const int row_in8 = row & 7;
  const int chunk16 = pair_in_atom >> 2;
  const int word_in_chunk = pair_in_atom & 3;
  return k16_atom * 1024 + row_group8 * 64 + chunk16 * 32 + row_in8 * 4 +
         word_in_chunk;
}

__host__ __device__ __forceinline__ uint64_t make_validation_smem_desc(
    uint32_t matrix_start_addr) {
  const uint32_t matrix_start_aligned = matrix_start_addr & ~0xFu;
  constexpr uint32_t leading_dim_byte_offset = 128;
  constexpr uint32_t stride_dim_byte_offset = 256;
  constexpr uint32_t swizzle_mode = 0;  // SWIZZLE_NONE.
  const uint32_t lead_enc = (leading_dim_byte_offset & 0x3ffffu) >> 4;
  const uint32_t stride_enc = (stride_dim_byte_offset & 0x3ffffu) >> 4;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>(lead_enc) << 16;
  desc |= static_cast<uint64_t>(stride_enc) << 32;
  desc |= static_cast<uint64_t>(0x1u) << 46;
  desc |= static_cast<uint64_t>(0xB0u) << 53;
  desc |= static_cast<uint64_t>(swizzle_mode) << 61;
  return desc;
}

float logical_value(ValidationPattern pattern, char matrix, int tile, int row, int col) {
  (void)tile;
  (void)col;
  if (pattern == ValidationPattern::Constant) {
    if (matrix == 'q') return kQValue;
    if (matrix == 'k') return kKValue;
    return kVValue;
  }
  if (pattern == ValidationPattern::Onehot) {
    if (matrix == 'q') {
      const bool row_match = row == g_probe_q_row;
      const bool col_match = g_probe_q_col < 0 || col == g_probe_q_col;
      return row_match && col_match ? kQValue : 0.0f;
    }
    if (matrix == 'k') return kKValue;
    return kVValue;
  }
  if (matrix == 'q') {
    return 0.00390625f * static_cast<float>(1 + (row % 13));
  }
  if (matrix == 'k') {
    return 0.00390625f * static_cast<float>(1 + (row % 11));
  }
  return 0.0078125f * static_cast<float>(1 + (row % 7)) +
         0.00048828125f * static_cast<float>(col % 17);
}

void fill_logical_matrix(std::vector<uint32_t>& words,
                         ValidationPattern pattern,
                         char matrix,
                         int tiles) {
  std::fill(words.begin(), words.end(), 0);
  for (int tile = 0; tile < tiles; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const float lo = bf16_bits_to_float(
            float_to_bf16_bits(matrix == 'v'
                                   ? logical_value(pattern, matrix, tile, 2 * col_pair, row)
                                   : logical_value(pattern, matrix, tile, row, 2 * col_pair)));
        const float hi = bf16_bits_to_float(
            float_to_bf16_bits(matrix == 'v'
                                   ? logical_value(pattern, matrix, tile, 2 * col_pair + 1, row)
                                   : logical_value(pattern, matrix, tile, row, 2 * col_pair + 1)));
        words[static_cast<size_t>(tile) * kTileWords +
              logical_word_offset(row, col_pair)] = pack_bf16_pair(lo, hi);
      }
    }
  }
}

__device__ __forceinline__ uint16_t val_float_to_bf16_bits(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ uint32_t val_pack_bf16_pair(float lo, float hi) {
  return static_cast<uint32_t>(val_float_to_bf16_bits(lo)) |
         (static_cast<uint32_t>(val_float_to_bf16_bits(hi)) << 16);
}

#define VAL_ST_GLOBAL_R64                                                      \
  "st.global.b32 [addr + 0], r0; "                                            \
  "st.global.b32 [addr + 4], r1; "                                            \
  "st.global.b32 [addr + 8], r2; "                                            \
  "st.global.b32 [addr + 12], r3; "                                           \
  "st.global.b32 [addr + 16], r4; "                                           \
  "st.global.b32 [addr + 20], r5; "                                           \
  "st.global.b32 [addr + 24], r6; "                                           \
  "st.global.b32 [addr + 28], r7; "                                           \
  "st.global.b32 [addr + 32], r8; "                                           \
  "st.global.b32 [addr + 36], r9; "                                           \
  "st.global.b32 [addr + 40], r10; "                                          \
  "st.global.b32 [addr + 44], r11; "                                          \
  "st.global.b32 [addr + 48], r12; "                                          \
  "st.global.b32 [addr + 52], r13; "                                          \
  "st.global.b32 [addr + 56], r14; "                                          \
  "st.global.b32 [addr + 60], r15; "                                          \
  "st.global.b32 [addr + 64], r16; "                                          \
  "st.global.b32 [addr + 68], r17; "                                          \
  "st.global.b32 [addr + 72], r18; "                                          \
  "st.global.b32 [addr + 76], r19; "                                          \
  "st.global.b32 [addr + 80], r20; "                                          \
  "st.global.b32 [addr + 84], r21; "                                          \
  "st.global.b32 [addr + 88], r22; "                                          \
  "st.global.b32 [addr + 92], r23; "                                          \
  "st.global.b32 [addr + 96], r24; "                                          \
  "st.global.b32 [addr + 100], r25; "                                         \
  "st.global.b32 [addr + 104], r26; "                                         \
  "st.global.b32 [addr + 108], r27; "                                         \
  "st.global.b32 [addr + 112], r28; "                                         \
  "st.global.b32 [addr + 116], r29; "                                         \
  "st.global.b32 [addr + 120], r30; "                                         \
  "st.global.b32 [addr + 124], r31; "                                         \
  "st.global.b32 [addr + 128], r32; "                                         \
  "st.global.b32 [addr + 132], r33; "                                         \
  "st.global.b32 [addr + 136], r34; "                                         \
  "st.global.b32 [addr + 140], r35; "                                         \
  "st.global.b32 [addr + 144], r36; "                                         \
  "st.global.b32 [addr + 148], r37; "                                         \
  "st.global.b32 [addr + 152], r38; "                                         \
  "st.global.b32 [addr + 156], r39; "                                         \
  "st.global.b32 [addr + 160], r40; "                                         \
  "st.global.b32 [addr + 164], r41; "                                         \
  "st.global.b32 [addr + 168], r42; "                                         \
  "st.global.b32 [addr + 172], r43; "                                         \
  "st.global.b32 [addr + 176], r44; "                                         \
  "st.global.b32 [addr + 180], r45; "                                         \
  "st.global.b32 [addr + 184], r46; "                                         \
  "st.global.b32 [addr + 188], r47; "                                         \
  "st.global.b32 [addr + 192], r48; "                                         \
  "st.global.b32 [addr + 196], r49; "                                         \
  "st.global.b32 [addr + 200], r50; "                                         \
  "st.global.b32 [addr + 204], r51; "                                         \
  "st.global.b32 [addr + 208], r52; "                                         \
  "st.global.b32 [addr + 212], r53; "                                         \
  "st.global.b32 [addr + 216], r54; "                                         \
  "st.global.b32 [addr + 220], r55; "                                         \
  "st.global.b32 [addr + 224], r56; "                                         \
  "st.global.b32 [addr + 228], r57; "                                         \
  "st.global.b32 [addr + 232], r58; "                                         \
  "st.global.b32 [addr + 236], r59; "                                         \
  "st.global.b32 [addr + 240], r60; "                                         \
  "st.global.b32 [addr + 244], r61; "                                         \
  "st.global.b32 [addr + 248], r62; "                                         \
  "st.global.b32 [addr + 252], r63; "

__device__ __forceinline__ void dump_tmem_x64_logical(uint32_t src_taddr,
                                                      float* tile_out,
                                                      int row_block,
                                                      int col_base,
                                                      int lane) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float* dst = tile_out + static_cast<size_t>(row_block * 32 + lane) * kTileN + col_base;
  asm volatile(
      "{ .reg .b32 r<64>; .reg .u64 addr; mov.u64 addr, %0; "
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TMEM_REGS_0_63 "}, [%1]; "
      "tcgen05.wait::ld.sync.aligned; "
      VAL_ST_GLOBAL_R64
      "}"
      :
      : "l"(dst), "r"(src_taddr)
      : "memory");
#else
  (void)src_taddr;
  (void)tile_out;
  (void)row_block;
  (void)col_base;
  (void)lane;
#endif
}

__global__ __launch_bounds__(kValidationThreads, 1)
void qk_validate_kernel(const __grid_constant__ CUtensorMap q_map,
                        const __grid_constant__ CUtensorMap k_map,
                        float* p_out) {
  extern __shared__ __align__(1024) unsigned char smem[];
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem);
  uint32_t* k_smem = reinterpret_cast<uint32_t*>(smem + kTileBytes);
  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready;
  __shared__ uint64_t qk_done;
  __shared__ uint32_t tmem_smem;

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
    mbarrier_init(&k_ready, 1);
    mbarrier_init(&qk_done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  uint32_t taddr = 0;
  if (warp_id == 0) {
    taddr = tcgen05_alloc_512cols(&tmem_smem);
  }
  __syncthreads();
  taddr = tmem_smem;
  tcgen05_fence_after_thread_sync();
  __syncthreads();

  if (threadIdx.x == 0) {
    mbarrier_expect_tx(&q_ready, kTileBytes);
    tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0, 0);
    mbarrier_expect_tx(&k_ready, kTileBytes);
    tma_load_4d(&k_map, smem_ptr_u32(k_smem), &k_ready, 0, 0,
                static_cast<int>(blockIdx.x) * 64, 0);
  }
  mbarrier_wait(&q_ready, 0);
  mbarrier_wait(&k_ready, 0);

  if (warp_id == 0 && lane == 0) {
    const uint32_t idesc = make_qk_idesc();
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      const uint64_t q_desc =
          make_validation_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      const uint64_t k_desc =
          make_validation_smem_desc(smem_ptr_u32(k_smem + mma * 1024));
      tcgen05_mma_bf16_ss(taddr, q_desc, k_desc, idesc, mma != 0);
    }
    tcgen05_commit(&qk_done);
  }
  mbarrier_wait(&qk_done, 0);

  if (warp_id < 4) {
    float* tile_out = p_out + static_cast<size_t>(blockIdx.x) * kTileBf16Elems;
    const uint32_t row_taddr = taddr + (static_cast<uint32_t>(warp_id * 32) << 16);
    dump_tmem_x64_logical(row_taddr, tile_out, warp_id, 0, lane);
    dump_tmem_x64_logical(row_taddr + 64u, tile_out, warp_id, 64, lane);
  }
  __syncthreads();
  if (warp_id == 0) {
    tcgen05_dealloc_512cols(taddr);
    tcgen05_relinquish_alloc_permit();
  }
}

__global__ void softmax_validate_kernel(const float* p,
                                        const float* row_max,
                                        uint32_t* s_words,
                                        float* row_sum,
                                        int k_tiles,
                                        float softmax_m,
                                        bool use_row_max) {
  const size_t total_words = static_cast<size_t>(k_tiles) * kTileWords;
  const size_t word_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (word_idx >= total_words) {
    return;
  }
  const int tile = static_cast<int>(word_idx / kTileWords);
  const int inner = static_cast<int>(word_idx % kTileWords);
  const int row = inner / (kTileN / 2);
  const int col_pair = inner % (kTileN / 2);
  const size_t elem0 =
      static_cast<size_t>(tile) * kTileBf16Elems + row * kTileN + col_pair * 2;
  const float max_value = use_row_max ? row_max[row] : softmax_m;
  const float e0 = exp2f(p[elem0] - max_value);
  const float e1 = exp2f(p[elem0 + 1] - max_value);
  s_words[static_cast<size_t>(tile) * kTileWords + logical_word_offset(row, col_pair)] =
      val_pack_bf16_pair(e0, e1);
  atomicAdd(row_sum + row, e0 + e1);
}

__global__ void row_max_validate_kernel(const float* p, float* row_max, int k_tiles) {
  __shared__ float partial[256];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int total_cols = k_tiles * kTileN;
  float max_value = -3.402823466e+38F;
  for (int col = tid; col < total_cols; col += blockDim.x) {
    const int tile = col / kTileN;
    const int tile_col = col - tile * kTileN;
    const size_t idx =
        static_cast<size_t>(tile) * kTileBf16Elems + row * kTileN + tile_col;
    max_value = fmaxf(max_value, p[idx]);
  }
  partial[tid] = max_value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) row_max[row] = partial[0];
}

__global__ __launch_bounds__(kValidationThreads, 1)
void pv_validate_kernel(const __grid_constant__ CUtensorMap s_map,
                        const __grid_constant__ CUtensorMap v_map,
                        float* o_out,
                        int k_tiles) {
  extern __shared__ __align__(1024) unsigned char smem[];
  uint32_t* s_smem = reinterpret_cast<uint32_t*>(smem);
  uint32_t* v_smem = reinterpret_cast<uint32_t*>(smem + kTileBytes);
  __shared__ uint64_t s_ready;
  __shared__ uint64_t v_ready;
  __shared__ uint64_t pv_done;
  __shared__ uint32_t tmem_smem;

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    mbarrier_init(&s_ready, 1);
    mbarrier_init(&v_ready, 1);
    mbarrier_init(&pv_done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  uint32_t taddr = 0;
  if (warp_id == 0) {
    taddr = tcgen05_alloc_512cols(&tmem_smem);
  }
  __syncthreads();
  taddr = tmem_smem;
  tcgen05_fence_after_thread_sync();
  __syncthreads();

  const uint32_t idesc = make_qk_idesc();
  for (int tile = 0; tile < k_tiles; ++tile) {
    if (threadIdx.x == 0) {
      mbarrier_expect_tx(&s_ready, kTileBytes);
      tma_load_4d(&s_map, smem_ptr_u32(s_smem), &s_ready, 0, 0, tile * 64, 0);
      mbarrier_expect_tx(&v_ready, kTileBytes);
      tma_load_4d(&v_map, smem_ptr_u32(v_smem), &v_ready, 0, 0, tile * 64, 0);
    }
    mbarrier_wait(&s_ready, static_cast<uint32_t>(tile & 1));
    mbarrier_wait(&v_ready, static_cast<uint32_t>(tile & 1));
    if (warp_id == 0 && lane == 0) {
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        const uint64_t s_desc =
            make_validation_smem_desc(smem_ptr_u32(s_smem + mma * 1024));
        const uint64_t v_desc =
            make_validation_smem_desc(smem_ptr_u32(v_smem + mma * 1024));
        tcgen05_mma_bf16_ss(taddr, s_desc, v_desc, idesc, tile != 0 || mma != 0);
      }
      tcgen05_commit(&pv_done);
    }
    mbarrier_wait(&pv_done, static_cast<uint32_t>(tile & 1));
  }

  if (warp_id < 4) {
    const uint32_t row_taddr = taddr + (static_cast<uint32_t>(warp_id * 32) << 16);
    dump_tmem_x64_logical(row_taddr, o_out, warp_id, 0, lane);
    dump_tmem_x64_logical(row_taddr + 64u, o_out, warp_id, 64, lane);
  }
  __syncthreads();
  if (warp_id == 0) {
    tcgen05_dealloc_512cols(taddr);
    tcgen05_relinquish_alloc_permit();
  }
}

__global__ void normalize_kernel(const float* o_acc,
                                 const float* row_sum,
                                 float* o_norm,
                                 size_t elems) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elems) {
    return;
  }
  const int row = static_cast<int>((idx % kTileBf16Elems) / kTileN);
  o_norm[idx] = o_acc[idx] / row_sum[row];
}

CompareResult compare_vector(const std::string& stage,
                             const std::vector<float>& got,
                             const std::vector<float>& expected,
                             float abs_tol,
                             float rel_tol,
                             int dump_mismatch) {
  CompareResult result;
  result.stage = stage;
  int dumped = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float abs_err = std::fabs(got[i] - expected[i]);
    const float rel_err = abs_err / std::max(std::fabs(expected[i]), 1.0e-6f);
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);
    if (abs_err > abs_tol && rel_err > rel_tol) {
      if (result.bad_count == 0) {
        result.first_bad_row = static_cast<int>((i % kTileBf16Elems) / kTileN);
        result.first_bad_col = static_cast<int>(i % kTileN);
      }
      ++result.bad_count;
      if (dumped < dump_mismatch) {
        std::printf("%s mismatch[%zu] got=%g expected=%g abs=%g rel=%g\n",
                    stage.c_str(), i, got[i], expected[i], abs_err, rel_err);
        ++dumped;
      }
    }
  }
  result.ok = result.bad_count == 0;
  return result;
}

CompareResult compare_softmax(const std::vector<uint32_t>& s_words,
                              const std::vector<float>& row_sum,
                              const std::vector<float>& expected_e,
                              const std::vector<float>& expected_row_sum,
                              int k_tiles,
                              int dump_mismatch) {
  CompareResult result;
  result.stage = "softmax";
  int dumped = 0;
  const auto check_value = [&](size_t logical_idx, float got, float expected) {
    const float abs_err = std::fabs(got - expected);
    const float rel_err = abs_err / std::max(std::fabs(expected), 1.0e-6f);
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);
    if (rel_err > kSoftmaxRelTol && abs_err > 1.0e-5f) {
      if (result.bad_count == 0) {
        result.first_bad_row = static_cast<int>((logical_idx % kTileBf16Elems) / kTileN);
        result.first_bad_col = static_cast<int>(logical_idx % kTileN);
      }
      ++result.bad_count;
      if (dumped < dump_mismatch) {
        std::printf("softmax mismatch[%zu] got=%g expected=%g abs=%g rel=%g\n",
                    logical_idx, got, expected, abs_err, rel_err);
        ++dumped;
      }
    }
  };
  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const uint32_t packed =
            s_words[static_cast<size_t>(tile) * kTileWords + logical_word_offset(row, col_pair)];
        const size_t elem = static_cast<size_t>(tile) * kTileBf16Elems + row * kTileN +
                            col_pair * 2;
        check_value(elem, bf16_bits_to_float(static_cast<uint16_t>(packed & 0xffffu)),
                    expected_e[elem]);
        check_value(elem + 1, bf16_bits_to_float(static_cast<uint16_t>(packed >> 16)),
                    expected_e[elem + 1]);
      }
    }
  }
  for (int row = 0; row < kTileM; ++row) {
    check_value(static_cast<size_t>(row) * kTileN, row_sum[row], expected_row_sum[row]);
  }
  result.ok = result.bad_count == 0;
  return result;
}

std::vector<uint32_t> pack_row_major_bf16_words(const std::vector<float>& values) {
  std::vector<uint32_t> words(kTileWords, 0);
  for (int row = 0; row < kTileM; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const int col = col_pair * 2;
      words[row * (kTileN / 2) + col_pair] =
          pack_bf16_pair(values[row * kTileN + col], values[row * kTileN + col + 1]);
    }
  }
  return words;
}

std::vector<float> unpack_row_major_bf16_words(const std::vector<uint32_t>& words) {
  std::vector<float> values(kTileBf16Elems, 0.0f);
  for (int row = 0; row < kTileM; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const uint32_t packed = words[row * (kTileN / 2) + col_pair];
      const int col = col_pair * 2;
      values[row * kTileN + col] =
          bf16_bits_to_float(static_cast<uint16_t>(packed & 0xffffu));
      values[row * kTileN + col + 1] =
          bf16_bits_to_float(static_cast<uint16_t>(packed >> 16));
    }
  }
  return values;
}

CompareResult compare_bf16_words(const std::string& stage,
                                 const std::vector<uint32_t>& got,
                                 const std::vector<uint32_t>& expected,
                                 int dump_mismatch) {
  CompareResult result;
  result.stage = stage;
  int dumped = 0;
  for (size_t word = 0; word < got.size(); ++word) {
    if (got[word] == expected[word]) continue;
    const int row = static_cast<int>(word / (kTileN / 2));
    const int col_pair = static_cast<int>(word % (kTileN / 2));
    for (int lane = 0; lane < 2; ++lane) {
      const uint16_t got_bits =
          static_cast<uint16_t>(lane == 0 ? (got[word] & 0xffffu) : (got[word] >> 16));
      const uint16_t exp_bits =
          static_cast<uint16_t>(lane == 0 ? (expected[word] & 0xffffu)
                                          : (expected[word] >> 16));
      const float got_f = bf16_bits_to_float(got_bits);
      const float exp_f = bf16_bits_to_float(exp_bits);
      const float abs_err = std::fabs(got_f - exp_f);
      const float rel_err = abs_err / std::max(std::fabs(exp_f), 1.0e-6f);
      result.max_abs = std::max(result.max_abs, abs_err);
      result.max_rel = std::max(result.max_rel, rel_err);
      if (got_bits != exp_bits) {
        if (result.bad_count == 0) {
          result.first_bad_row = row;
          result.first_bad_col = col_pair * 2 + lane;
        }
        ++result.bad_count;
        if (dumped < dump_mismatch) {
          std::printf("%s mismatch[row=%d col=%d] got_bits=0x%04x expected_bits=0x%04x "
                      "got=%g expected=%g\n",
                      stage.c_str(), row, col_pair * 2 + lane, got_bits, exp_bits, got_f,
                      exp_f);
          ++dumped;
        }
      }
    }
  }
  result.ok = result.bad_count == 0;
  return result;
}

void write_probe_csv(const std::string& path,
                     const std::vector<float>& p,
                     int expected_row,
                     int expected_col) {
  std::ofstream out(path);
  out << "probe,logical_row,logical_col,dump_row,dump_col,value\n";
  for (int tile = 0; tile < 1; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      for (int col = 0; col < kTileN; ++col) {
        const float value = p[static_cast<size_t>(tile) * kTileBf16Elems + row * kTileN + col];
        if (std::fabs(value) > 1.0e-6f) {
          out << "qk_onehot," << expected_row << ',' << expected_col << ',' << row << ','
              << col << ',' << value << '\n';
        }
      }
    }
  }
}

void write_csv(const std::string& path, const std::vector<CompareResult>& results) {
  std::ofstream out(path);
  out << "stage,status,max_abs,max_rel,bad_count,first_bad_row,first_bad_col\n";
  for (const CompareResult& r : results) {
    out << r.stage << ',' << (r.ok ? "ok" : "fail") << ',' << r.max_abs << ','
        << r.max_rel << ',' << r.bad_count << ',' << r.first_bad_row << ','
        << r.first_bad_col << '\n';
  }
}

void encode_validation_atom_tma_map(CUtensorMap* map,
                                    void* base,
                                    uint64_t physical_rows) {
  constexpr uint64_t kAtomWords = 32;
  constexpr uint64_t kAtomsPerPhysicalRow = 4;
  const cuuint64_t global_dim[4] = {kAtomWords, kAtomsPerPhysicalRow, physical_rows, 1};
  const cuuint64_t global_stride[3] = {
      kAtomWords * sizeof(uint32_t),
      kAtomWords * kAtomsPerPhysicalRow * sizeof(uint32_t),
      kAtomWords * kAtomsPerPhysicalRow * physical_rows * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {static_cast<cuuint32_t>(kAtomWords),
                                 static_cast<cuuint32_t>(kAtomsPerPhysicalRow), 64, 1};
  const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                      4,
                                      base,
                                      global_dim,
                                      global_stride,
                                      box_dim,
                                      elem_stride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(validation_atom)");
}

bool needs_qk(const std::string& stage) {
  return stage == "qk" || stage == "softmax" || stage == "pv" || stage == "all" ||
         stage == "probe";
}

bool needs_softmax(const std::string& stage) {
  return stage == "softmax" || stage == "pv" || stage == "all";
}

bool needs_pv(const std::string& stage) {
  return stage == "pv" || stage == "all";
}

bool needs_fused(const std::string& stage) {
  return stage == "fused" || stage == "all";
}

void unpack_logical_matrix(const std::vector<uint32_t>& words,
                           int tiles,
                           std::vector<float>* values) {
  values->assign(static_cast<size_t>(tiles) * kTileBf16Elems, 0.0f);
  for (int tile = 0; tile < tiles; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const uint32_t packed =
            words[static_cast<size_t>(tile) * kTileWords + logical_word_offset(row, col_pair)];
        const size_t elem =
            static_cast<size_t>(tile) * kTileBf16Elems + row * kTileN + col_pair * 2;
        (*values)[elem] = bf16_bits_to_float(static_cast<uint16_t>(packed & 0xffffu));
        (*values)[elem + 1] = bf16_bits_to_float(static_cast<uint16_t>(packed >> 16));
      }
    }
  }
}

void build_references(const std::vector<uint32_t>& h_q,
                      const std::vector<uint32_t>& h_k,
                      const std::vector<uint32_t>& h_v,
                      int k_tiles,
                      float softmax_m,
                      bool use_row_max,
                      std::vector<float>* p_ref,
                      std::vector<float>* row_max_ref,
                      std::vector<float>* e_ref,
                      std::vector<float>* row_sum_ref,
                      std::vector<float>* o_ref,
                      std::vector<float>* norm_ref) {
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  unpack_logical_matrix(h_q, 1, &q);
  unpack_logical_matrix(h_k, k_tiles, &k);
  unpack_logical_matrix(h_v, k_tiles, &v);
  p_ref->assign(static_cast<size_t>(k_tiles) * kTileBf16Elems, 0.0f);
  row_max_ref->assign(kTileM, softmax_m);
  e_ref->assign(static_cast<size_t>(k_tiles) * kTileBf16Elems, 0.0f);
  row_sum_ref->assign(kTileM, 0.0f);
  o_ref->assign(kTileBf16Elems, 0.0f);
  norm_ref->assign(kTileBf16Elems, 0.0f);

  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int m = 0; m < kTileM; ++m) {
      for (int n = 0; n < kTileN; ++n) {
        float acc = 0.0f;
        for (int d = 0; d < kTileN; ++d) {
          acc += q[m * kTileN + d] *
                 k[static_cast<size_t>(tile) * kTileBf16Elems + n * kTileN + d];
        }
        const size_t idx = static_cast<size_t>(tile) * kTileBf16Elems + m * kTileN + n;
        (*p_ref)[idx] = acc;
        if (use_row_max && (tile == 0 && n == 0 || acc > (*row_max_ref)[m])) {
          (*row_max_ref)[m] = acc;
        }
      }
    }
  }

  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int m = 0; m < kTileM; ++m) {
      const float max_value = use_row_max ? (*row_max_ref)[m] : softmax_m;
      for (int n = 0; n < kTileN; ++n) {
        const size_t idx = static_cast<size_t>(tile) * kTileBf16Elems + m * kTileN + n;
        const float e = std::exp2((*p_ref)[idx] - max_value);
        (*e_ref)[idx] = e;
        (*row_sum_ref)[m] += e;
      }
    }
  }

  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int m = 0; m < kTileM; ++m) {
      for (int d = 0; d < kTileN; ++d) {
        float acc = (*o_ref)[m * kTileN + d];
        for (int n = 0; n < kTileN; ++n) {
          const size_t e_idx = static_cast<size_t>(tile) * kTileBf16Elems + m * kTileN + n;
          const size_t v_idx = static_cast<size_t>(tile) * kTileBf16Elems + d * kTileN + n;
          acc += (*e_ref)[e_idx] * v[v_idx];
        }
        (*o_ref)[m * kTileN + d] = acc;
      }
    }
  }

  for (int m = 0; m < kTileM; ++m) {
    for (int d = 0; d < kTileN; ++d) {
      (*norm_ref)[m * kTileN + d] = (*o_ref)[m * kTileN + d] / (*row_sum_ref)[m];
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const ValidationArgs args = parse_validation_args(argc, argv);
  g_probe_q_row = args.probe_row;
  g_probe_q_col = args.probe_col;
  CUDA_CHECK(cudaFree(nullptr));
  driver_check(cuInit(0), "cuInit");

  const size_t q_words = kTileWords;
  const size_t kv_words = static_cast<size_t>(args.k_tiles) * kTileWords;
  const size_t p_elems = static_cast<size_t>(args.k_tiles) * kTileBf16Elems;
  const size_t o_elems = kTileBf16Elems;

  std::vector<uint32_t> h_q(q_words);
  std::vector<uint32_t> h_k(kv_words);
  std::vector<uint32_t> h_v(kv_words);
  fill_logical_matrix(h_q, args.pattern, 'q', 1);
  fill_logical_matrix(h_k, args.pattern, 'k', args.k_tiles);
  fill_logical_matrix(h_v, args.pattern, 'v', args.k_tiles);

  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_s = nullptr;
  uint32_t* d_sink = nullptr;
  float* d_p = nullptr;
  float* d_o = nullptr;
  float* d_o_norm = nullptr;
  uint32_t* d_fused_o = nullptr;
  float* d_row_max = nullptr;
  float* d_row_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, kv_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, kv_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_s, kv_words * sizeof(uint32_t)));
  if (needs_fused(args.stage)) {
#if ATTENTION_STORE_OUTPUT
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_fused_o, kTileWords * sizeof(uint32_t)));
#else
    std::fprintf(stderr,
                 "--stage fused/all requires -DATTENTION_STORE_OUTPUT=1 "
                 "-DATTENTION_NVCC_MANAGED_LD_REGS=1\n");
    return 2;
#endif
  }
  CUDA_CHECK(cudaMalloc(&d_p, p_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, o_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o_norm, o_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_max, kTileM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_sum, kTileM * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_p, 0, p_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_o, 0, o_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_o_norm, 0, o_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_row_max, 0, kTileM * sizeof(float)));
  if (d_sink) CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint32_t)));
  if (d_fused_o) {
    CUDA_CHECK(cudaMemset(d_fused_o, 0, kTileWords * sizeof(uint32_t)));
  }
  CUDA_CHECK(cudaMemset(d_s, 0, kv_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_row_sum, 0, kTileM * sizeof(float)));

  CUtensorMap q_map{};
  CUtensorMap k_map{};
  CUtensorMap v_map{};
  CUtensorMap s_map{};
  CUtensorMap o_map{};
  encode_validation_atom_tma_map(&q_map, d_q, 64);
  encode_validation_atom_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_validation_atom_tma_map(&v_map, d_v, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_validation_atom_tma_map(&s_map, d_s, static_cast<uint64_t>(args.k_tiles) * 64);
  if (d_fused_o) {
    encode_bf16_output_tma_map(&o_map, d_fused_o, 1);
  }

  CUDA_CHECK(cudaFuncSetAttribute(qk_validate_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kValidationSmemBytes));
  CUDA_CHECK(cudaFuncSetAttribute(pv_validate_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kValidationSmemBytes));
#if ATTENTION_STORE_OUTPUT
  if (needs_fused(args.stage)) {
    CUDA_CHECK(cudaFuncSetAttribute(qk_tma_mma_ld_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kDynamicSmemBytes));
  }
#endif

  std::vector<CompareResult> results;
  std::vector<float> p_ref;
  std::vector<float> row_max_ref;
  std::vector<float> e_ref;
  std::vector<float> row_sum_ref;
  std::vector<float> o_ref;
  std::vector<float> norm_ref;
  build_references(h_q, h_k, h_v, args.k_tiles, args.softmax_m, args.row_max, &p_ref,
                   &row_max_ref, &e_ref, &row_sum_ref, &o_ref, &norm_ref);

  if (needs_qk(args.stage)) {
    qk_validate_kernel<<<args.k_tiles, kValidationThreads, kValidationSmemBytes>>>(q_map, k_map,
                                                                                   d_p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_p(p_elems);
    CUDA_CHECK(cudaMemcpy(h_p.data(), d_p, p_elems * sizeof(float), cudaMemcpyDeviceToHost));
    if (args.stage == "probe") {
      write_probe_csv(args.csv, h_p, args.probe_row, args.probe_col);
      std::printf("probe wrote %s\n", args.csv.c_str());
      return 0;
    }
    CompareResult qk =
        compare_vector("qk", h_p, p_ref, kQkAbsTol, 3.0e-2f, args.dump_mismatch);
    results.push_back(qk);
    if (!qk.ok && args.stage == "qk") {
      write_csv(args.csv, results);
      return 1;
    }
  }

  if (needs_softmax(args.stage)) {
    CUDA_CHECK(cudaMemset(d_row_sum, 0, kTileM * sizeof(float)));
    if (args.row_max) {
      row_max_validate_kernel<<<kTileM, 256>>>(d_p, d_row_max, args.k_tiles);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    const int threads = 256;
    const int blocks = static_cast<int>((kv_words + threads - 1) / threads);
    softmax_validate_kernel<<<blocks, threads>>>(d_p, d_row_max, d_s, d_row_sum,
                                                 args.k_tiles, args.softmax_m,
                                                 args.row_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> h_s(kv_words);
    std::vector<float> h_row_max(kTileM);
    std::vector<float> h_row_sum(kTileM);
    CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, kv_words * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_max.data(), d_row_max, kTileM * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_sum.data(), d_row_sum, kTileM * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (args.row_max) {
      results.push_back(compare_vector("row_max", h_row_max, row_max_ref, 0.0f, 0.0f,
                                       args.dump_mismatch));
    }
    CompareResult softmax =
        compare_softmax(h_s, h_row_sum, e_ref, row_sum_ref, args.k_tiles, args.dump_mismatch);
    results.push_back(softmax);
    if (!softmax.ok && args.stage == "softmax") {
      write_csv(args.csv, results);
      return 1;
    }
  }

  if (needs_pv(args.stage)) {
    pv_validate_kernel<<<1, kValidationThreads, kValidationSmemBytes>>>(s_map, v_map, d_o,
                                                                        args.k_tiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const int threads = 256;
    const int blocks = static_cast<int>((o_elems + threads - 1) / threads);
    normalize_kernel<<<blocks, threads>>>(d_o, d_row_sum, d_o_norm, o_elems);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_o(o_elems);
    std::vector<float> h_o_norm(o_elems);
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_o_norm.data(), d_o_norm, o_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    results.push_back(compare_vector("pv", h_o, o_ref, 2.0e-2f, kPvRelTol,
                                     args.dump_mismatch));
    results.push_back(compare_vector("norm", h_o_norm, norm_ref, 1.0e-3f, kPvRelTol,
                                     args.dump_mismatch));
  }

  if (needs_fused(args.stage)) {
#if ATTENTION_STORE_OUTPUT
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_fused_o, 0, kTileWords * sizeof(uint32_t)));
    qk_tma_mma_ld_kernel<<<1, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, o_map, d_sink, args.k_tiles, args.k_tiles, d_fused_o);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_fused(kTileWords, 0);
    CUDA_CHECK(cudaMemcpy(h_fused.data(), d_fused_o, kTileWords * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    const std::vector<float> h_fused_norm = unpack_row_major_bf16_words(h_fused);
    const std::vector<uint32_t> expected_bf16 = pack_row_major_bf16_words(norm_ref);
    results.push_back(compare_vector("fused_norm", h_fused_norm, norm_ref, 1.0e-3f,
                                     kPvRelTol, args.dump_mismatch));
    results.push_back(compare_bf16_words("final_o_bf16", h_fused, expected_bf16,
                                         args.dump_mismatch));
#endif
  }

  write_csv(args.csv, results);
  bool all_ok = true;
  for (const CompareResult& r : results) {
    all_ok = all_ok && r.ok;
    std::printf("%s status=%s max_abs=%g max_rel=%g bad=%zu first=(%d,%d)\n",
                r.stage.c_str(), r.ok ? "ok" : "fail", r.max_abs, r.max_rel,
                r.bad_count, r.first_bad_row, r.first_bad_col);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_s));
  if (d_sink) CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_o));
  CUDA_CHECK(cudaFree(d_o_norm));
  if (d_fused_o) CUDA_CHECK(cudaFree(d_fused_o));
  CUDA_CHECK(cudaFree(d_row_max));
  CUDA_CHECK(cudaFree(d_row_sum));
  return all_ok ? 0 : 1;
}
