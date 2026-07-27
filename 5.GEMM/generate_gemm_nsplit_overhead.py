#!/usr/bin/env python3
import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new)


def apply_no_memset(text: str) -> str:
    return replace_once(
        text,
        "    cuda_check(cudaMemsetAsync(d_sink + ctas, 0, sizeof(uint32_t)));\n",
        "",
        "per-launch sink-counter memset",
    )


def apply_sinkless(text: str, trim_barriers: bool) -> str:
    text = replace_once(
        text,
        "  __shared__ uint32_t warp_sinks[kWarps];\n",
        "",
        "warp_sinks declaration",
    )
    text = replace_once(
        text,
        """    const uint32_t acc = static_cast<uint32_t>(threadIdx.x + 0x9e3779b9u);
    if (lane0)
      warp_sinks[warp_id] = acc;
    __syncthreads();

""",
        "" if trim_barriers else "    __syncthreads();\n\n",
        "warp sink publication",
    )
    text = replace_once(
        text,
        """    __syncthreads();

    if (threadIdx.x == 0) {
      uint32_t tile_sink = tmem_base ^ static_cast<uint32_t>(ktiles);
#pragma unroll
      for (int w = 0; w < kWarps; ++w)
        tile_sink ^= warp_sinks[w];
      sink[tile_m * ntile + tile_n] = tile_sink;
    }
    __syncthreads();

""",
        (
            ""
            if trim_barriers
            else """    __syncthreads();
    __syncthreads();

"""
        ),
        "tile sink store",
    )
    return apply_no_memset(text)


def apply_fixed_16k(text: str) -> str:
    text = replace_once(
        text,
        """__global__ __launch_bounds__(kThreads, 1) void gemm256_bf16_16k_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap c_map, uint32_t *__restrict__ sink,
    int ktiles, int mtile_count, int ntile_count) {
""",
        """template <int ktiles, int mtile_count, int ntile_count>
__global__ __launch_bounds__(kThreads, 1) void gemm256_bf16_16k_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap c_map, uint32_t *__restrict__ sink) {
""",
        "kernel template",
    )
    text = replace_once(
        text,
        """void set_gemm_kernel_attribute() {
  cuda_check(cudaFuncSetAttribute(gemm256_bf16_16k_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
}

void launch_gemm_kernel(dim3 grid, const CUtensorMap &a_map,
                        const CUtensorMap &b_map, const CUtensorMap &c_map,
                        uint32_t *d_sink, int ktiles, int mtile, int ntile) {
  gemm256_bf16_16k_kernel<<<grid, kThreads, kDynamicSmemBytes>>>(
      a_map, b_map, c_map, d_sink, ktiles, mtile, ntile);
}
""",
        """template <int KTiles, int MTiles, int NTiles>
void set_one_gemm_kernel_attribute() {
  cuda_check(cudaFuncSetAttribute(
      gemm256_bf16_16k_kernel<KTiles, MTiles, NTiles>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes));
}

void set_gemm_kernel_attribute() {
  set_one_gemm_kernel_attribute<256, 64, 64>();
  set_one_gemm_kernel_attribute<8, 2, 2>();
  set_one_gemm_kernel_attribute<4, 1, 1>();
}

template <int KTiles, int MTiles, int NTiles>
void launch_one_gemm_kernel(dim3 grid, const CUtensorMap &a_map,
                            const CUtensorMap &b_map,
                            const CUtensorMap &c_map, uint32_t *d_sink) {
  gemm256_bf16_16k_kernel<KTiles, MTiles, NTiles>
      <<<grid, kThreads, kDynamicSmemBytes>>>(a_map, b_map, c_map, d_sink);
}

void launch_gemm_kernel(dim3 grid, const CUtensorMap &a_map,
                        const CUtensorMap &b_map, const CUtensorMap &c_map,
                        uint32_t *d_sink, int ktiles, int mtile, int ntile) {
  if (ktiles == 256 && mtile == 64 && ntile == 64) {
    launch_one_gemm_kernel<256, 64, 64>(grid, a_map, b_map, c_map, d_sink);
  } else if (ktiles == 8 && mtile == 2 && ntile == 2) {
    launch_one_gemm_kernel<8, 2, 2>(grid, a_map, b_map, c_map, d_sink);
  } else if (ktiles == 4 && mtile == 1 && ntile == 1) {
    launch_one_gemm_kernel<4, 1, 1>(grid, a_map, b_map, c_map, d_sink);
  } else {
    std::fprintf(stderr, "Unsupported specialized shape: %d/%d/%d\\n",
                 ktiles, mtile, ntile);
    std::exit(EXIT_FAILURE);
  }
}
""",
        "specialized launch",
    )
    return text


def apply_suspend(text: str) -> str:
    wait_end = """#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t *barrier,
"""
    suspend_helper = """#endif
}

__device__ __forceinline__ void mbarrier_wait_suspend(uint64_t *barrier,
                                                      uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  constexpr uint32_t kSuspendTicks = 0x989680u;
  asm volatile("{ .reg .pred p; "
               "L_wait_%=: "
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
               "@p bra.uni L_done_%=; "
               "bra.uni L_wait_%=; "
               "L_done_%=: }" ::"r"(addr),
               "r"(phase), "r"(kSuspendTicks)
               : "memory");
#else
  (void)barrier;
  (void)phase;
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t *barrier,
"""
    text = replace_once(text, wait_end, suspend_helper, "suspend helper")
    text = replace_once(
        text,
        "            mbarrier_wait(&mma_done[p][stage], reuse_phase);\n",
        "            mbarrier_wait_suspend(&mma_done[p][stage], reuse_phase);\n",
        "warp-0 producer wait",
    )
    text = replace_once(
        text,
        """          mbarrier_wait(
              &mma_done[1][stage],
""",
        """          mbarrier_wait_suspend(
              &mma_done[1][stage],
""",
        "warp-1 producer wait",
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sink",
        choices=("baseline", "no-memset", "keep-sync", "trim"),
        default="baseline",
    )
    parser.add_argument("--fixed-16k", action="store_true")
    parser.add_argument("--suspend-producer", action="store_true")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    text = Path(args.base).read_text()
    if args.sink == "no-memset":
        text = apply_no_memset(text)
    elif args.sink == "keep-sync":
        text = apply_sinkless(text, trim_barriers=False)
    elif args.sink == "trim":
        text = apply_sinkless(text, trim_barriers=True)
    if args.fixed_16k:
        text = apply_fixed_16k(text)
    if args.suspend_producer:
        text = apply_suspend(text)
    text = text.replace(
        "scheduler=static_%dx%d_mfast ",
        f"scheduler=static_%dx%d_mfast overhead={args.label} ",
    )
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
