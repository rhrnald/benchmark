#pragma once

// Low-level PTX and Blackwell wrapper helpers for the attention kernel.

#ifndef ATTENTION_SETMAXNREG_PRODUCER
#define ATTENTION_SETMAXNREG_PRODUCER 120
#endif

#ifndef ATTENTION_SETMAXNREG_CONSUMER
#define ATTENTION_SETMAXNREG_CONSUMER 192
#endif

#define ATTENTION_STRINGIFY_IMPL(x) #x
#define ATTENTION_STRINGIFY(x) ATTENTION_STRINGIFY_IMPL(x)

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__host__ __device__ __forceinline__ uint64_t make_smem_desc(uint32_t matrix_start_addr) {
  constexpr uint32_t leading_dim_byte_offset = 128;
  constexpr uint32_t stride_dim_byte_offset = 256;
  constexpr uint32_t swizzle_mode = 0;
  constexpr uint64_t lead_enc =
      static_cast<uint64_t>((leading_dim_byte_offset & 0x3ffffu) >> 4);
  constexpr uint64_t stride_enc =
      static_cast<uint64_t>((stride_dim_byte_offset & 0x3ffffu) >> 4);
  constexpr uint64_t desc_base = (lead_enc << 16) | (stride_enc << 32) |
                                 (static_cast<uint64_t>(0x1u) << 46) |
                                 (static_cast<uint64_t>(0xB0u) << 53) |
                                 (static_cast<uint64_t>(swizzle_mode) << 61);
  return desc_base | static_cast<uint64_t>((matrix_start_addr & ~0xFu) >> 4);
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_k_smem_desc_addr16(
    uint32_t matrix_start_addr16,
    int mma) {
  constexpr uint64_t desc_base = (static_cast<uint64_t>(1u) << 16) |   // leading byte offset = 16B.
                                 (static_cast<uint64_t>(64u) << 32) |  // stride byte offset = 1024B.
                                 (static_cast<uint64_t>(1u) << 46) |   // Blackwell descriptor version.
                                 (static_cast<uint64_t>(2u) << 61);    // SWIZZLE_128B.
  const int half = mma >> 2;
  const int in_half = mma & 3;
  const uint32_t addr16 =
      matrix_start_addr16 + static_cast<uint32_t>(half * ((kTileBytes / 2) >> 4) +
                                                  in_half * (32 >> 4));
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_k_smem_desc(
    uint32_t matrix_start_addr,
    int mma) {
  return make_sw128_major_k_smem_desc_addr16((matrix_start_addr & ~0xFu) >> 4, mma);
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_mn_smem_desc_addr16(
    uint32_t matrix_start_addr16,
    int mma) {
  constexpr uint64_t desc_base = (static_cast<uint64_t>(128u) << 16) |  // leading byte offset = 2048B.
                                 (static_cast<uint64_t>(64u) << 32) |   // stride byte offset = 1024B.
                                 (static_cast<uint64_t>(1u) << 46) |    // Blackwell descriptor version.
                                 (static_cast<uint64_t>(2u) << 61);     // SWIZZLE_128B.
  const uint32_t addr16 = matrix_start_addr16 + static_cast<uint32_t>(mma) * (4096u >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_mn_smem_desc(
    uint32_t matrix_start_addr,
    int mma) {
  return make_sw128_major_mn_smem_desc_addr16((matrix_start_addr & ~0xFu) >> 4, mma);
}

__host__ __device__ __forceinline__ uint64_t make_s_smem_desc_addr16(
    uint32_t matrix_start_addr16) {
  constexpr uint32_t leading_dim_byte_offset = 128;
  constexpr uint32_t stride_dim_byte_offset = 256;
  constexpr uint64_t lead_enc =
      static_cast<uint64_t>((leading_dim_byte_offset & 0x3ffffu) >> 4);
  constexpr uint64_t stride_enc =
      static_cast<uint64_t>((stride_dim_byte_offset & 0x3ffffu) >> 4);
  constexpr uint64_t desc_base = (lead_enc << 16) | (stride_enc << 32) |
                                 (static_cast<uint64_t>(0x1u) << 46) |
                                 (static_cast<uint64_t>(0xB0u) << 53);
  return desc_base | static_cast<uint64_t>(matrix_start_addr16);
}

__host__ __device__ __forceinline__ uint64_t make_s_smem_desc(uint32_t matrix_start_addr) {
  return make_s_smem_desc_addr16((matrix_start_addr & ~0xFu) >> 4);
}

__host__ __device__ __forceinline__ int atom_major_k_word_offset(int row, int col_pair) {
  const int k16_atom = col_pair >> 3;
  const int pair_in_atom = col_pair & 7;
  const int row_group8 = row >> 3;
  const int row_in8 = row & 7;
  const int chunk16 = pair_in_atom >> 2;
  const int word_in_chunk = pair_in_atom & 3;
  return k16_atom * 1024 + row_group8 * 64 + chunk16 * 32 + row_in8 * 4 +
         word_in_chunk;
}

__host__ __device__ __forceinline__ uint32_t swizzle_s_byte_offset(uint32_t byte_offset) {
  return byte_offset;
}

__host__ __device__ __forceinline__ int s_store_word_offset(int row, int col_pair) {
  const uint32_t byte_offset =
      static_cast<uint32_t>(atom_major_k_word_offset(row, col_pair)) * 4u;
  return static_cast<int>(swizzle_s_byte_offset(byte_offset) >> 2);
}

__host__ __device__ __forceinline__ uint32_t make_qk_idesc() {
  uint32_t desc = 0;
  desc |= 1u << 4;   // C format: F32.
  desc |= 1u << 7;   // A format: BF16.
  desc |= 1u << 10;  // B format: BF16.
  desc |= static_cast<uint32_t>(kTileN >> 3) << 17;
  desc |= static_cast<uint32_t>(kTileM >> 4) << 24;
  return desc;
}

__device__ __forceinline__ void setmaxnreg_dec_producer() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("setmaxnreg.dec.sync.aligned.u32 "
               ATTENTION_STRINGIFY(ATTENTION_SETMAXNREG_PRODUCER) ";"
               ::: "memory");
#endif
}

__device__ __forceinline__ void setmaxnreg_inc_consumer() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 "
               ATTENTION_STRINGIFY(ATTENTION_SETMAXNREG_CONSUMER) ";"
               ::: "memory");
#endif
}

__device__ __forceinline__ bool warp_elect_leader() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t is_leader = 0;
  asm volatile(
      "{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }"
      : "=r"(is_leader)
      :
      : "memory");
  return is_leader != 0;
#else
  return (threadIdx.x & 31) == 0;
#endif
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count)
               : "memory");
#else
  (void)barrier;
  (void)count;
#endif
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile(
      "{ .reg .pred p; "
      "L_wait_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
      "@p bra.uni L_done_%=; "
      "bra.uni L_wait_%=; "
      "L_done_%=: }"
      :: "r"(addr), "r"(phase)
      : "memory");
#else
  (void)barrier;
  (void)phase;
#endif
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
#else
  (void)barrier;
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* barrier, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("{ .reg .pred p; "
               "elect.sync _|p, -1; "
               "@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1; }"
               :: "r"(addr), "r"(bytes)
               : "memory");
#else
  (void)barrier;
  (void)bytes;
#endif
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map,
                                            uint32_t dst_smem,
                                            uint64_t* barrier,
                                            int c,
                                            int r) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(dst_smem), "l"(map), "r"(bar), "r"(c), "r"(r)
      : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c;
  (void)r;
#endif
}

__device__ __forceinline__ void issue_k_tma_tile(const CUtensorMap* k_map,
                                                 uint32_t* dst_smem,
                                                 uint64_t* barrier,
                                                 int global_k_tile,
                                                 bool lane0) {
  mbarrier_expect_tx(barrier, kTileBytes);
  if (lane0) {
    const uint32_t dst_smem_addr = smem_ptr_u32(dst_smem);
    tma_load_2d(k_map, dst_smem_addr, barrier, 0, global_k_tile * kTileM);
    tma_load_2d(k_map, dst_smem_addr + kTileBytes / 2, barrier, 32,
                global_k_tile * kTileM);
  }
}

__device__ __forceinline__ void tma_load_4d(const CUtensorMap* map,
                                            uint32_t dst_smem,
                                            uint64_t* barrier,
                                            int c,
                                            int r,
                                            int d,
                                            int b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];"
      :
      : "r"(dst_smem), "l"(map), "r"(bar), "r"(c), "r"(r), "r"(d), "r"(b)
      : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c;
  (void)r;
  (void)d;
  (void)b;
#endif
}

__device__ __forceinline__ void issue_v_tma_tile(const CUtensorMap* v_map,
                                                 uint32_t* dst_smem,
                                                 uint64_t* barrier,
                                                 int global_v_tile,
                                                 bool lane0) {
  mbarrier_expect_tx(barrier, kTileBytes);
  if (lane0) {
    const uint32_t dst_smem_addr = smem_ptr_u32(dst_smem);
    tma_load_4d(v_map, dst_smem_addr, barrier, 0, 0, 0, global_v_tile * 8);
  }
}

__device__ __forceinline__ void tma_load_5d(const CUtensorMap* map,
                                            uint32_t dst_smem,
                                            uint64_t* barrier,
                                            int c0,
                                            int c1,
                                            int c2,
                                            int c3,
                                            int c4) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
      :
      : "r"(dst_smem), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2),
        "r"(c3), "r"(c4)
      : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c0;
  (void)c1;
  (void)c2;
  (void)c3;
  (void)c4;
#endif
}

__device__ __forceinline__ void tma_store_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_4d(const CUtensorMap* map,
                                             uint32_t src_smem,
                                             int c,
                                             int r,
                                             int d,
                                             int b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
      " [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(map), "r"(src_smem), "r"(c), "r"(r), "r"(d), "r"(b)
      : "memory");
#else
  (void)map;
  (void)src_smem;
  (void)c;
  (void)r;
  (void)d;
  (void)b;
#endif
}

__device__ __forceinline__ void tma_store_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_wait_group_read() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
#endif
}

__device__ __forceinline__ uint32_t tcgen05_alloc_512cols(uint32_t* smem_out_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t smem_addr = smem_ptr_u32(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
               :: "r"(smem_addr)
               : "memory");
  __syncwarp();
  uint32_t taddr;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr) : "r"(smem_addr) : "memory");
  return taddr;
#else
  (void)smem_out_taddr;
  return 0;
#endif
}

__device__ __forceinline__ void tcgen05_dealloc_512cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;"
               :: "r"(taddr)
               : "memory");
#else
  (void)taddr;
#endif
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_mma_bf16_ss(uint32_t d_taddr,
                                                    uint64_t a_desc,
                                                    uint64_t b_desc,
                                                    uint32_t idesc,
                                                    bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile(
      "{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, pred; }"
      :: "r"(d_taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p),
         "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
      : "memory");
#else
  (void)d_taddr;
  (void)a_desc;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

__device__ __forceinline__ void tcgen05_commit(uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(addr)
               : "memory");
#else
  (void)barrier;
#endif
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_wait_st() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif
}

#define TCGEN05_LD_X64_OUTPUTS(a)                                            \
  "=&r"(a[0]), "=&r"(a[1]), "=&r"(a[2]), "=&r"(a[3]), "=&r"(a[4]),       \
      "=&r"(a[5]), "=&r"(a[6]), "=&r"(a[7]), "=&r"(a[8]), "=&r"(a[9]),    \
      "=&r"(a[10]), "=&r"(a[11]), "=&r"(a[12]), "=&r"(a[13]),             \
      "=&r"(a[14]), "=&r"(a[15]), "=&r"(a[16]), "=&r"(a[17]),             \
      "=&r"(a[18]), "=&r"(a[19]), "=&r"(a[20]), "=&r"(a[21]),             \
      "=&r"(a[22]), "=&r"(a[23]), "=&r"(a[24]), "=&r"(a[25]),             \
      "=&r"(a[26]), "=&r"(a[27]), "=&r"(a[28]), "=&r"(a[29]),             \
      "=&r"(a[30]), "=&r"(a[31]), "=&r"(a[32]), "=&r"(a[33]),             \
      "=&r"(a[34]), "=&r"(a[35]), "=&r"(a[36]), "=&r"(a[37]),             \
      "=&r"(a[38]), "=&r"(a[39]), "=&r"(a[40]), "=&r"(a[41]),             \
      "=&r"(a[42]), "=&r"(a[43]), "=&r"(a[44]), "=&r"(a[45]),             \
      "=&r"(a[46]), "=&r"(a[47]), "=&r"(a[48]), "=&r"(a[49]),             \
      "=&r"(a[50]), "=&r"(a[51]), "=&r"(a[52]), "=&r"(a[53]),             \
      "=&r"(a[54]), "=&r"(a[55]), "=&r"(a[56]), "=&r"(a[57]),             \
      "=&r"(a[58]), "=&r"(a[59]), "=&r"(a[60]), "=&r"(a[61]),             \
      "=&r"(a[62]), "=&r"(a[63])

#define TCGEN05_LD_X64_OPERANDS                                              \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "  \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "  \
  "%30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "  \
  "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "  \
  "%58, %59, %60, %61, %62, %63"

#define TCGEN05_LD_X64(src_taddr, out_regs)                                \
  asm volatile(                                                            \
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TCGEN05_LD_X64_OPERANDS   \
      "}, [%64];"                                                          \
      : TCGEN05_LD_X64_OUTPUTS(out_regs)                                   \
      : "r"(src_taddr)                                                    \
      : "memory")

#define TCGEN05_LD_X32_OUTPUTS(a)                                            \
  "=&r"(a[0]), "=&r"(a[1]), "=&r"(a[2]), "=&r"(a[3]), "=&r"(a[4]),       \
      "=&r"(a[5]), "=&r"(a[6]), "=&r"(a[7]), "=&r"(a[8]), "=&r"(a[9]),    \
      "=&r"(a[10]), "=&r"(a[11]), "=&r"(a[12]), "=&r"(a[13]),             \
      "=&r"(a[14]), "=&r"(a[15]), "=&r"(a[16]), "=&r"(a[17]),             \
      "=&r"(a[18]), "=&r"(a[19]), "=&r"(a[20]), "=&r"(a[21]),             \
      "=&r"(a[22]), "=&r"(a[23]), "=&r"(a[24]), "=&r"(a[25]),             \
      "=&r"(a[26]), "=&r"(a[27]), "=&r"(a[28]), "=&r"(a[29]),             \
      "=&r"(a[30]), "=&r"(a[31])

#define TCGEN05_LD_X32_OPERANDS                                              \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "  \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "  \
  "%30, %31"

#define TCGEN05_LD_X32(src_taddr, out_regs)                                \
  asm volatile(                                                            \
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" TCGEN05_LD_X32_OPERANDS   \
      "}, [%32];"                                                          \
      : TCGEN05_LD_X32_OUTPUTS(out_regs)                                   \
      : "r"(src_taddr)                                                    \
      : "memory")

#define TCGEN05_ST_X32_INPUTS(a)                                             \
  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]),   \
      "r"(a[6]), "r"(a[7]), "r"(a[8]), "r"(a[9]), "r"(a[10]),          \
      "r"(a[11]), "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),       \
      "r"(a[16]), "r"(a[17]), "r"(a[18]), "r"(a[19]), "r"(a[20]),       \
      "r"(a[21]), "r"(a[22]), "r"(a[23]), "r"(a[24]), "r"(a[25]),       \
      "r"(a[26]), "r"(a[27]), "r"(a[28]), "r"(a[29]), "r"(a[30]),       \
      "r"(a[31])

#define TCGEN05_ST_X32_OPERANDS                                              \
  "%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, " \
  "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "  \
  "%31, %32"

#define TCGEN05_ST_X32(dst_taddr, in_regs)                                  \
  asm volatile(                                                             \
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {"                      \
      TCGEN05_ST_X32_OPERANDS "};"                                          \
      :                                                                     \
      : "r"(dst_taddr), TCGEN05_ST_X32_INPUTS(in_regs)                     \
      : "memory")

#define TCGEN05_LD_X16_OUTPUTS(a)                                            \
  "=&r"(a[0]), "=&r"(a[1]), "=&r"(a[2]), "=&r"(a[3]), "=&r"(a[4]),       \
      "=&r"(a[5]), "=&r"(a[6]), "=&r"(a[7]), "=&r"(a[8]), "=&r"(a[9]),    \
      "=&r"(a[10]), "=&r"(a[11]), "=&r"(a[12]), "=&r"(a[13]),             \
      "=&r"(a[14]), "=&r"(a[15])

#define TCGEN05_LD_X16_OPERANDS                                              \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15"

#define TCGEN05_LD_X16(src_taddr, out_regs)                                \
  asm volatile(                                                            \
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" TCGEN05_LD_X16_OPERANDS   \
      "}, [%16];"                                                          \
      : TCGEN05_LD_X16_OUTPUTS(out_regs)                                   \
      : "r"(src_taddr)                                                    \
      : "memory")

__device__ __forceinline__ uint32_t exp2_approx_bits_cpp(uint32_t x) {
  const float in = __uint_as_float(x);
  float out;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(in));
  return __float_as_uint(out);
}

__device__ __forceinline__ float exp2_approx_float_cpp(float x) {
  float out;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}

__device__ __forceinline__ uint32_t exp2_approx_bits_scaled_cpp(uint32_t x,
                                                                float scale) {
  const float in = __uint_as_float(x) * scale;
  float out;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(in));
  return __float_as_uint(out);
}

__device__ __forceinline__ uint32_t exp2_approx_bits_scaled_shifted_cpp(
    uint32_t x,
    float scale,
    float row_max_shift) {
  const float in = __uint_as_float(x) * scale - row_max_shift;
  float out;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(in));
  return __float_as_uint(out);
}

__device__ __forceinline__ void exp2_emulation_2_bits_scaled_cpp(
    uint32_t& lo_src,
    uint32_t& hi_src,
    float scale) {
  const float x = __uint_as_float(lo_src) * scale;
  const float y = __uint_as_float(hi_src) * scale;
  uint32_t out_x;
  uint32_t out_y;
  asm volatile(
      "{\n\t"
      ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
      ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
      ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
      "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
      "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
      "mov.b64 l1, {f1, f2};\n\t"
      "mov.f32 f3, 0f4B400000;\n\t"
      "mov.b64 l2, {f3, f3};\n\t"
      "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
      "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
      "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
      "mov.f32 f7, 0f3D9DF09D;\n\t"
      "mov.b64 l6, {f7, f7};\n\t"
      "mov.f32 f6, 0f3E6906A4;\n\t"
      "mov.b64 l5, {f6, f6};\n\t"
      "mov.f32 f5, 0f3F31F519;\n\t"
      "mov.b64 l4, {f5, f5};\n\t"
      "mov.f32 f4, 0f3F800000;\n\t"
      "mov.b64 l3, {f4, f4};\n\t"
      "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
      "mov.b64 {r1, r2}, l7;\n\t"
      "mov.b64 {r3, r4}, l10;\n\t"
      "shl.b32 r5, r1, 23;\n\t"
      "add.s32 r7, r5, r3;\n\t"
      "shl.b32 r6, r2, 23;\n\t"
      "add.s32 r8, r6, r4;\n\t"
      "mov.b32 %0, r7;\n\t"
      "mov.b32 %1, r8;\n\t"
      "}\n"
      : "=r"(out_x), "=r"(out_y)
      : "f"(x), "f"(y));
  lo_src = out_x;
  hi_src = out_y;
}

__device__ __forceinline__ void exp2_emulation_2_bits_scaled_shifted_cpp(
    uint32_t& lo_src,
    uint32_t& hi_src,
    float scale,
    float row_max_shift) {
  const float x = __uint_as_float(lo_src) * scale - row_max_shift;
  const float y = __uint_as_float(hi_src) * scale - row_max_shift;
  uint32_t out_x;
  uint32_t out_y;
  asm volatile(
      "{\n\t"
      ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
      ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
      ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
      "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
      "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
      "mov.b64 l1, {f1, f2};\n\t"
      "mov.f32 f3, 0f4B400000;\n\t"
      "mov.b64 l2, {f3, f3};\n\t"
      "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
      "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
      "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
      "mov.f32 f7, 0f3D9DF09D;\n\t"
      "mov.b64 l6, {f7, f7};\n\t"
      "mov.f32 f6, 0f3E6906A4;\n\t"
      "mov.b64 l5, {f6, f6};\n\t"
      "mov.f32 f5, 0f3F31F519;\n\t"
      "mov.b64 l4, {f5, f5};\n\t"
      "mov.f32 f4, 0f3F800000;\n\t"
      "mov.b64 l3, {f4, f4};\n\t"
      "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
      "mov.b64 {r1, r2}, l7;\n\t"
      "mov.b64 {r3, r4}, l10;\n\t"
      "shl.b32 r5, r1, 23;\n\t"
      "add.s32 r7, r5, r3;\n\t"
      "shl.b32 r6, r2, 23;\n\t"
      "add.s32 r8, r6, r4;\n\t"
      "mov.b32 %0, r7;\n\t"
      "mov.b32 %1, r8;\n\t"
      "}\n"
      : "=r"(out_x), "=r"(out_y)
      : "f"(x), "f"(y));
  lo_src = out_x;
  hi_src = out_y;
}

__device__ __forceinline__ uint32_t exp2_pack_hi16_update(uint32_t& lo_src,
                                                          uint32_t& hi_src) {
  lo_src = exp2_approx_bits_cpp(lo_src);
  hi_src = exp2_approx_bits_cpp(hi_src);
  return (lo_src >> 16) | (hi_src & 0xffff0000u);
}

__device__ __forceinline__ uint32_t exp2_pack_hi16_update_scaled(
    uint32_t& lo_src,
    uint32_t& hi_src,
    float scale,
    int pair_index) {
  if ((pair_index % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
    exp2_emulation_2_bits_scaled_cpp(lo_src, hi_src, scale);
  } else {
    lo_src = exp2_approx_bits_scaled_cpp(lo_src, scale);
    hi_src = exp2_approx_bits_scaled_cpp(hi_src, scale);
  }
  return (lo_src >> 16) | (hi_src & 0xffff0000u);
}

__device__ __forceinline__ uint32_t exp2_pack_hi16_update_scaled_shifted(
    uint32_t& lo_src,
    uint32_t& hi_src,
    float scale,
    float row_max_shift,
    int pair_index) {
  if ((pair_index % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
    exp2_emulation_2_bits_scaled_shifted_cpp(lo_src, hi_src, scale,
                                             row_max_shift);
  } else {
    lo_src = exp2_approx_bits_scaled_shifted_cpp(lo_src, scale,
                                                 row_max_shift);
    hi_src = exp2_approx_bits_scaled_shifted_cpp(hi_src, scale,
                                                 row_max_shift);
  }
  return (lo_src >> 16) | (hi_src & 0xffff0000u);
}

__device__ __forceinline__ float bf16x2_sum_device(uint32_t packed) {
  const float lo = __uint_as_float((packed & 0x0000ffffu) << 16);
  const float hi = __uint_as_float(packed & 0xffff0000u);
  return lo + hi;
}

__device__ __forceinline__ void store_packed4_s_cpp(uint32_t* smem,
                                                    int word_offset,
                                                    uint32_t p0,
                                                    uint32_t p1,
                                                    uint32_t p2,
                                                    uint32_t p3) {
  reinterpret_cast<uint4*>(smem + word_offset)[0] = make_uint4(p0, p1, p2, p3);
}

template <bool kDoSum>
__device__ __forceinline__ float pack_store_x64_loop(uint32_t* smem_base,
                                                     uint32_t (&r)[64],
                                                     float score_to_exp2_scale) {
  float sum = 0.0f;
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    alignas(16) uint32_t p[4];
    const int r_base = group * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = exp2_pack_hi16_update_scaled(r[r_base + i * 2],
                                          r[r_base + i * 2 + 1],
                                          score_to_exp2_scale,
                                          group * 4 + i);
    }
    const int word_offset = (group >> 1) * 1024 + (group & 1) * 32;
    reinterpret_cast<uint4*>(smem_base + word_offset)[0] =
        reinterpret_cast<uint4*>(p)[0];
    if constexpr (kDoSum) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sum += bf16x2_sum_device(p[i]);
      }
    }
  }
  return sum;
}

__device__ __forceinline__ float row_max_x64_scaled(uint32_t (&r)[64],
                                                    float score_to_exp2_scale) {
  float row_max = -3.4028234663852886e+38f;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    row_max = fmaxf(row_max, __uint_as_float(r[i]) * score_to_exp2_scale);
  }
  return row_max;
}

template <bool kDoSum>
__device__ __forceinline__ float pack_store_x64_loop_shifted(
    uint32_t* smem_base,
    uint32_t (&r)[64],
    float score_to_exp2_scale,
    float row_max_shift) {
  float sum = 0.0f;
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    alignas(16) uint32_t p[4];
    const int r_base = group * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = exp2_pack_hi16_update_scaled_shifted(
          r[r_base + i * 2], r[r_base + i * 2 + 1], score_to_exp2_scale,
          row_max_shift, group * 4 + i);
    }
    const int word_offset = (group >> 1) * 1024 + (group & 1) * 32;
    reinterpret_cast<uint4*>(smem_base + word_offset)[0] =
        reinterpret_cast<uint4*>(p)[0];
    if constexpr (kDoSum) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sum += bf16x2_sum_device(p[i]);
      }
    }
  }
  return sum;
}

struct PackStoreX64LoopResult {
  float sum;
  float row_max;
};

template <bool kDoSum, bool kDoMax>
__device__ __forceinline__ PackStoreX64LoopResult
pack_store_x64_loop_result(uint32_t* smem_base,
                           uint32_t (&r)[64],
                           float score_to_exp2_scale) {
  float sum = 0.0f;
  float row_max = -3.4028234663852886e+38f;
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    alignas(16) uint32_t p[4];
    const int r_base = group * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (kDoMax) {
        const float lo =
            __uint_as_float(r[r_base + i * 2]) * score_to_exp2_scale;
        const float hi =
            __uint_as_float(r[r_base + i * 2 + 1]) * score_to_exp2_scale;
        row_max = fmaxf(row_max, fmaxf(lo, hi));
      }
      p[i] = exp2_pack_hi16_update_scaled(r[r_base + i * 2],
                                          r[r_base + i * 2 + 1],
                                          score_to_exp2_scale,
                                          group * 4 + i);
    }
    const int word_offset = (group >> 1) * 1024 + (group & 1) * 32;
    reinterpret_cast<uint4*>(smem_base + word_offset)[0] =
        reinterpret_cast<uint4*>(p)[0];
    if constexpr (kDoSum) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sum += bf16x2_sum_device(p[i]);
      }
    }
  }
  return {sum, row_max};
}

#if ATTENTION_CLOCK_TRACE
__device__ __forceinline__ void write_clock_trace_record(ClockTraceRecord* records,
                                                         int slot,
                                                         int stage,
                                                         int iter,
                                                         int pipe,
                                                         int warp_id,
                                                         int consumer_warp,
                                                         int half,
                                                         unsigned long long start,
                                                         unsigned long long end,
                                                         unsigned long long base);

template <bool kDoSum>
__device__ __forceinline__ float pack_store_x64_loop_trace_detail(
    uint32_t* smem_base,
    uint32_t (&r)[64],
    float score_to_exp2_scale,
    ClockTraceRecord* clock_trace,
    int trace_slot_base,
    int trace_iter,
    int trace_pipe,
    int warp_id,
    int consumer_warp,
    int consumer_half,
    bool trace_lane,
    unsigned long long clock_trace_base) {
  float sum = 0.0f;
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    const bool trace_group =
        trace_lane && group == ATTENTION_CLOCK_TRACE_PACK_GROUP;
    const unsigned long long group_start = trace_group ? clock64() : 0ull;
    alignas(16) uint32_t p[4];
    const int r_base = group * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = exp2_pack_hi16_update_scaled(r[r_base + i * 2],
                                          r[r_base + i * 2 + 1],
                                          score_to_exp2_scale,
                                          group * 4 + i);
    }
    const int word_offset = (group >> 1) * 1024 + (group & 1) * 32;
    reinterpret_cast<uint4*>(smem_base + word_offset)[0] =
        reinterpret_cast<uint4*>(p)[0];
    if constexpr (kDoSum) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sum += bf16x2_sum_device(p[i]);
      }
    }
    if (trace_group) {
      const unsigned long long group_end = clock64();
      const int slot = trace_slot_base + kClockTracePackDetailBase +
                       consumer_warp * 2 + consumer_half;
      write_clock_trace_record(clock_trace, slot, kClockTracePackDetail,
                               trace_iter, trace_pipe, warp_id, consumer_warp,
                               consumer_half, group_start, group_end,
                               clock_trace_base);
    }
  }
  return sum;
}
#endif

__device__ __forceinline__ void write_clock_trace_record(ClockTraceRecord* records,
                                                         int slot,
                                                         int stage,
                                                         int iter,
                                                         int pipe,
                                                         int warp_id,
                                                         int consumer_warp,
                                                         int half,
                                                         unsigned long long start,
                                                         unsigned long long end,
                                                         unsigned long long base) {
#if ATTENTION_CLOCK_TRACE
  if (records == nullptr || end <= start) return;
  ClockTraceRecord r;
  r.stage = stage;
  r.iter = iter;
  r.pipe = pipe;
  r.warp_id = warp_id;
  r.consumer_warp = consumer_warp;
  r.half = half;
  r.start = start - base;
  r.end = end - base;
  records[slot] = r;
#else
  (void)records;
  (void)slot;
  (void)stage;
  (void)iter;
  (void)pipe;
  (void)warp_id;
  (void)consumer_warp;
  (void)half;
  (void)start;
  (void)end;
  (void)base;
#endif
}

__device__ __forceinline__ void begin_clock_trace_record(ClockTraceRecord* records,
                                                         int slot,
                                                         int stage,
                                                         int iter,
                                                         int pipe,
                                                         int warp_id,
                                                         int consumer_warp,
                                                         int half,
                                                         unsigned long long start,
                                                         unsigned long long base) {
#if ATTENTION_CLOCK_TRACE
  if (records == nullptr) return;
  ClockTraceRecord r;
  r.stage = stage;
  r.iter = iter;
  r.pipe = pipe;
  r.warp_id = warp_id;
  r.consumer_warp = consumer_warp;
  r.half = half;
  r.start = start - base;
  r.end = r.start;
  records[slot] = r;
#else
  (void)records;
  (void)slot;
  (void)stage;
  (void)iter;
  (void)pipe;
  (void)warp_id;
  (void)consumer_warp;
  (void)half;
  (void)start;
  (void)base;
#endif
}

__device__ __forceinline__ void end_clock_trace_record(ClockTraceRecord* records,
                                                       int slot,
                                                       unsigned long long end,
                                                       unsigned long long base) {
#if ATTENTION_CLOCK_TRACE
  if (records == nullptr) return;
  records[slot].end = end - base;
#else
  (void)records;
  (void)slot;
  (void)end;
  (void)base;
#endif
}



__device__ __forceinline__ float
tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
    uint32_t src_taddr,
    uint32_t* s_smem,
    int consumer_warp,
    int consumer_half,
    uint64_t* p_done_barrier,
    bool arrive_p_done,
    float score_to_exp2_scale,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    int trace_iter,
    int trace_pipe) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int row = consumer_warp * 32 + lane;
  const int col_pair_base = consumer_half * 32;
  uint32_t* smem_base = s_smem + s_store_word_offset(row, col_pair_base);
  uint32_t r[64];

#if ATTENTION_CLOCK_TRACE
  const int trace_idx = trace_iter - clock_trace_start;
  const bool trace_window =
      clock_trace != nullptr && blockIdx.x == 0 && trace_idx >= 0 &&
      trace_idx < clock_trace_iters;
  const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
  const bool trace_lane = trace_window && lane == 0;
  const unsigned long long ld_start = trace_lane ? clock64() : 0ull;
#endif
  TCGEN05_LD_X64(src_taddr, r);
  tcgen05_wait_ld();
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long ld_end = clock64();
    const int slot =
        trace_slot_base + kClockTraceLdBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, slot, kClockTraceLd, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, ld_start, ld_end, clock_trace_base);
  }
#endif
  if (arrive_p_done && lane == 0) {
    mbarrier_arrive(p_done_barrier);
  }

#if ATTENTION_CLOCK_TRACE
  const unsigned long long pack_start = trace_lane ? clock64() : 0ull;
#endif
#if ATTENTION_CLOCK_TRACE && ATTENTION_CLOCK_TRACE_PACK_GROUP >= 0
  const float row_sum = pack_store_x64_loop_trace_detail<true>(
      smem_base, r, score_to_exp2_scale, clock_trace, trace_slot_base,
      trace_iter, trace_pipe, threadIdx.x >> 5, consumer_warp, consumer_half,
      trace_lane, clock_trace_base);
#else
  const float row_sum = pack_store_x64_loop<true>(smem_base, r,
                                                 score_to_exp2_scale);
#endif
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long pack_end = clock64();
    const int pack_slot = trace_slot_base + kClockTracePackStoreBase +
                          consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, pack_slot, kClockTracePack, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int store_slot =
        trace_slot_base + kClockTraceStoreBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, store_slot, kClockTraceStore, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int sum_slot =
        trace_slot_base + kClockTraceRowSumBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, sum_slot, kClockTraceRowSum, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
  }
#endif
  return row_sum;
#else
  (void)src_taddr;
  (void)s_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)p_done_barrier;
  (void)arrive_p_done;
  (void)score_to_exp2_scale;
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
  (void)clock_trace_base;
  (void)trace_iter;
  (void)trace_pipe;
  return 0.0f;
#endif
}

__device__ __forceinline__ float tcgen05_ld_x64_wait_row_max_scaled_nvcc(
    uint32_t src_taddr,
    float score_to_exp2_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r[64];
  TCGEN05_LD_X64(src_taddr, r);
  tcgen05_wait_ld();
  return row_max_x64_scaled(r, score_to_exp2_scale);
#else
  (void)src_taddr;
  (void)score_to_exp2_scale;
  return -3.4028234663852886e+38f;
#endif
}

__device__ __forceinline__ float
tcgen05_ld_x64_wait_pack_store_sum_shift_half_nvcc(
    uint32_t src_taddr,
    uint32_t* s_smem,
    int consumer_warp,
    int consumer_half,
    uint64_t* p_done_barrier,
    bool arrive_p_done,
    float score_to_exp2_scale,
    float row_max_shift,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    int trace_iter,
    int trace_pipe) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int row = consumer_warp * 32 + lane;
  const int col_pair_base = consumer_half * 32;
  uint32_t* smem_base = s_smem + s_store_word_offset(row, col_pair_base);
  uint32_t r[64];

#if ATTENTION_CLOCK_TRACE
  const int trace_idx = trace_iter - clock_trace_start;
  const bool trace_window =
      clock_trace != nullptr && blockIdx.x == 0 && trace_idx >= 0 &&
      trace_idx < clock_trace_iters;
  const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
  const bool trace_lane = trace_window && lane == 0;
  const unsigned long long ld_start = trace_lane ? clock64() : 0ull;
#endif
  TCGEN05_LD_X64(src_taddr, r);
  tcgen05_wait_ld();
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long ld_end = clock64();
    const int slot =
        trace_slot_base + kClockTraceLdBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, slot, kClockTraceLd, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, ld_start, ld_end, clock_trace_base);
  }
#endif
  if (arrive_p_done && lane == 0) {
    mbarrier_arrive(p_done_barrier);
  }

#if ATTENTION_CLOCK_TRACE
  const unsigned long long pack_start = trace_lane ? clock64() : 0ull;
#endif
  const float row_sum = pack_store_x64_loop_shifted<true>(
      smem_base, r, score_to_exp2_scale, row_max_shift);
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long pack_end = clock64();
    const int pack_slot = trace_slot_base + kClockTracePackStoreBase +
                          consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, pack_slot, kClockTracePack, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int store_slot =
        trace_slot_base + kClockTraceStoreBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, store_slot, kClockTraceStore, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int sum_slot =
        trace_slot_base + kClockTraceRowSumBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, sum_slot, kClockTraceRowSum, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
  }
#endif
  return row_sum;
#else
  (void)src_taddr;
  (void)s_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)p_done_barrier;
  (void)arrive_p_done;
  (void)score_to_exp2_scale;
  (void)row_max_shift;
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
  (void)clock_trace_base;
  (void)trace_iter;
  (void)trace_pipe;
  return 0.0f;
#endif
}

__device__ __forceinline__ PackStoreX64LoopResult
tcgen05_ld_x64_wait_pack_store_sum_max_half_nvcc(
    uint32_t src_taddr,
    uint32_t* s_smem,
    int consumer_warp,
    int consumer_half,
    uint64_t* p_done_barrier,
    bool arrive_p_done,
    float score_to_exp2_scale,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    int trace_iter,
    int trace_pipe) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int row = consumer_warp * 32 + lane;
  const int col_pair_base = consumer_half * 32;
  uint32_t* smem_base = s_smem + s_store_word_offset(row, col_pair_base);
  uint32_t r[64];

#if ATTENTION_CLOCK_TRACE
  const int trace_idx = trace_iter - clock_trace_start;
  const bool trace_window =
      clock_trace != nullptr && blockIdx.x == 0 && trace_idx >= 0 &&
      trace_idx < clock_trace_iters;
  const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
  const bool trace_lane = trace_window && lane == 0;
  const unsigned long long ld_start = trace_lane ? clock64() : 0ull;
#endif
  TCGEN05_LD_X64(src_taddr, r);
  tcgen05_wait_ld();
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long ld_end = clock64();
    const int slot =
        trace_slot_base + kClockTraceLdBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, slot, kClockTraceLd, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, ld_start, ld_end, clock_trace_base);
  }
#endif
  if (arrive_p_done && lane == 0) {
    mbarrier_arrive(p_done_barrier);
  }

#if ATTENTION_CLOCK_TRACE
  const unsigned long long pack_start = trace_lane ? clock64() : 0ull;
#endif
  PackStoreX64LoopResult result =
      pack_store_x64_loop_result<true, true>(smem_base, r,
                                             score_to_exp2_scale);
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long pack_end = clock64();
    const int pack_slot = trace_slot_base + kClockTracePackStoreBase +
                          consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, pack_slot, kClockTracePack, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int store_slot =
        trace_slot_base + kClockTraceStoreBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, store_slot, kClockTraceStore, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int sum_slot =
        trace_slot_base + kClockTraceRowSumBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, sum_slot, kClockTraceRowSum, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
  }
#endif
  return result;
#else
  (void)src_taddr;
  (void)s_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)p_done_barrier;
  (void)arrive_p_done;
  (void)score_to_exp2_scale;
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
  (void)clock_trace_base;
  (void)trace_iter;
  (void)trace_pipe;
  return {0.0f, -3.4028234663852886e+38f};
#endif
}

__device__ __noinline__ void scale_tmem_x32_accum(uint32_t taddr,
                                                  float scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r[32];
  TCGEN05_LD_X32(taddr, r);
  tcgen05_wait_ld();
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    r[i] = __float_as_uint(__uint_as_float(r[i]) * scale);
  }
  TCGEN05_ST_X32(taddr, r);
  tcgen05_wait_st();
#else
  (void)taddr;
  (void)scale;
#endif
}

__device__ __noinline__ void scale_tmem_x128_accum(uint32_t row_block_taddr,
                                                   float scale) {
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    scale_tmem_x32_accum(row_block_taddr + static_cast<uint32_t>(chunk * 32),
                         scale);
  }
}


__device__ __forceinline__ float bf16_bits_to_float_device(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits_device(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ uint32_t pack_bf16_pair_device(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits_device(lo)) |
         (static_cast<uint32_t>(float_to_bf16_bits_device(hi)) << 16);
}

__device__ __noinline__ void store_tmem_x32_pair_norm_bf16_smem(
    uint32_t src0_taddr,
    uint32_t src1_taddr,
    uint32_t* dst_bf16_smem,
    float inv_sum) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r0[32];
  uint32_t r1[32];
  TCGEN05_LD_X32(src0_taddr, r0);
  TCGEN05_LD_X32(src1_taddr, r1);
  tcgen05_wait_ld();
#pragma unroll
  for (int i = 0; i < 32; i += 2) {
    const float lo =
        (__uint_as_float(r0[i]) + __uint_as_float(r1[i])) * inv_sum;
    const float hi =
        (__uint_as_float(r0[i + 1]) + __uint_as_float(r1[i + 1])) * inv_sum;
    dst_bf16_smem[i >> 1] = pack_bf16_pair_device(lo, hi);
  }
#else
  (void)src0_taddr;
  (void)src1_taddr;
  (void)dst_bf16_smem;
  (void)inv_sum;
#endif
}

__device__ __noinline__ void store_tmem_x32_pair_scale_norm_bf16_smem(
    uint32_t src0_taddr,
    uint32_t src1_taddr,
    uint32_t* dst_bf16_smem,
    float scale0,
    float scale1,
    float inv_sum) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r0[32];
  uint32_t r1[32];
  TCGEN05_LD_X32(src0_taddr, r0);
  TCGEN05_LD_X32(src1_taddr, r1);
  tcgen05_wait_ld();
#pragma unroll
  for (int i = 0; i < 32; i += 2) {
    const float lo =
        (__uint_as_float(r0[i]) * scale0 + __uint_as_float(r1[i]) * scale1) *
        inv_sum;
    const float hi =
        (__uint_as_float(r0[i + 1]) * scale0 +
         __uint_as_float(r1[i + 1]) * scale1) *
        inv_sum;
    dst_bf16_smem[i >> 1] = pack_bf16_pair_device(lo, hi);
  }
#else
  (void)src0_taddr;
  (void)src1_taddr;
  (void)dst_bf16_smem;
  (void)scale0;
  (void)scale1;
  (void)inv_sum;
#endif
}

__device__ __noinline__ void store_tmem_x32_norm_bf16_smem(
    uint32_t src_taddr,
    uint32_t* dst_bf16_smem,
    float inv_sum) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r[32];
  TCGEN05_LD_X32(src_taddr, r);
  tcgen05_wait_ld();
#pragma unroll
  for (int i = 0; i < 32; i += 2) {
    const float lo = __uint_as_float(r[i]) * inv_sum;
    const float hi = __uint_as_float(r[i + 1]) * inv_sum;
    dst_bf16_smem[i >> 1] = pack_bf16_pair_device(lo, hi);
  }
#else
  (void)src_taddr;
  (void)dst_bf16_smem;
  (void)inv_sum;
#endif
}

__device__ __noinline__ void store_tmem_x16_pair_norm_bf16_smem(
    uint32_t src0_taddr,
    uint32_t src1_taddr,
    uint32_t* dst_bf16_smem,
    float inv_sum) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r0[16];
  uint32_t r1[16];
  TCGEN05_LD_X16(src0_taddr, r0);
  TCGEN05_LD_X16(src1_taddr, r1);
  tcgen05_wait_ld();
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    const float lo =
        (__uint_as_float(r0[i]) + __uint_as_float(r1[i])) * inv_sum;
    const float hi =
        (__uint_as_float(r0[i + 1]) + __uint_as_float(r1[i + 1])) * inv_sum;
    dst_bf16_smem[i >> 1] = pack_bf16_pair_device(lo, hi);
  }
#else
  (void)src0_taddr;
  (void)src1_taddr;
  (void)dst_bf16_smem;
  (void)inv_sum;
#endif
}

__device__ __noinline__ void store_tmem_x16_pair_scale_norm_bf16_smem(
    uint32_t src0_taddr,
    uint32_t src1_taddr,
    uint32_t* dst_bf16_smem,
    float scale0,
    float scale1,
    float inv_sum) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r0[16];
  uint32_t r1[16];
  TCGEN05_LD_X16(src0_taddr, r0);
  TCGEN05_LD_X16(src1_taddr, r1);
  tcgen05_wait_ld();
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    const float lo =
        (__uint_as_float(r0[i]) * scale0 + __uint_as_float(r1[i]) * scale1) *
        inv_sum;
    const float hi =
        (__uint_as_float(r0[i + 1]) * scale0 +
         __uint_as_float(r1[i + 1]) * scale1) *
        inv_sum;
    dst_bf16_smem[i >> 1] = pack_bf16_pair_device(lo, hi);
  }
#else
  (void)src0_taddr;
  (void)src1_taddr;
  (void)dst_bf16_smem;
  (void)scale0;
  (void)scale1;
  (void)inv_sum;
#endif
}

__device__ __noinline__ void store_tmem_x16_norm_bf16_smem(
    uint32_t src_taddr,
    uint32_t* dst_bf16_smem,
    float inv_sum) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t r[16];
  TCGEN05_LD_X16(src_taddr, r);
  tcgen05_wait_ld();
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    const float lo = __uint_as_float(r[i]) * inv_sum;
    const float hi = __uint_as_float(r[i + 1]) * inv_sum;
    dst_bf16_smem[i >> 1] = pack_bf16_pair_device(lo, hi);
  }
#else
  (void)src_taddr;
  (void)dst_bf16_smem;
  (void)inv_sum;
#endif
}

__device__ __noinline__ void store_tmem_x64_accum_output_smem(uint32_t src_taddr,
                                                              float* output_smem,
                                                              int consumer_warp,
                                                              int consumer_half,
                                                              bool add_to_smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  float* dst = output_smem + static_cast<size_t>(consumer_warp * 32 + lane) * kTileN +
               consumer_half * 64;
  uint32_t r[64];
  TCGEN05_LD_X64(src_taddr, r);
  tcgen05_wait_ld();
  if (add_to_smem) {
#pragma unroll
    for (int i = 0; i < 64; i += 4) {
      const float4 prev = reinterpret_cast<float4*>(dst + i)[0];
      reinterpret_cast<float4*>(dst + i)[0] =
          make_float4(prev.x + __uint_as_float(r[i + 0]),
                      prev.y + __uint_as_float(r[i + 1]),
                      prev.z + __uint_as_float(r[i + 2]),
                      prev.w + __uint_as_float(r[i + 3]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < 64; i += 4) {
      reinterpret_cast<float4*>(dst + i)[0] =
          make_float4(__uint_as_float(r[i + 0]), __uint_as_float(r[i + 1]),
                      __uint_as_float(r[i + 2]), __uint_as_float(r[i + 3]));
    }
  }
#else
  (void)src_taddr;
  (void)output_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)add_to_smem;
#endif
}
