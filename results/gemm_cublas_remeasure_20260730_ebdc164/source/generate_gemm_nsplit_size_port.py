#!/usr/bin/env python3
import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}, got {count}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--macro-m", required=True, type=int)
    parser.add_argument("--macro-n", required=True, type=int)
    parser.add_argument(
        "--scheduler", choices=("dynamic", "static"), default="static"
    )
    args = parser.parse_args()

    if args.size not in (8192, 16384, 32768):
        raise ValueError("size must be 8192, 16384, or 32768")
    if args.macro_m <= 0 or args.macro_n <= 0:
        raise ValueError("macro dimensions must be positive")

    text = Path(args.base).read_text()
    text = replace_once(
        text,
        "static constexpr int kBenchmarkSize = 16384;",
        f"static constexpr int kBenchmarkSize = {args.size};",
    )
    performance_shapes = {
        8192: (128, 32, 32),
        16384: (256, 64, 64),
        32768: (512, 128, 128),
    }
    ktiles, mtiles, ntiles = performance_shapes[args.size]
    if args.size != 16384:
        text = replace_once(
            text,
            "  set_one_gemm_kernel_attribute<256, 64, 64>();",
            f"  set_one_gemm_kernel_attribute<{ktiles}, {mtiles}, {ntiles}>();",
        )
        text = replace_once(
            text,
            """  if (ktiles == 256 && mtile == 64 && ntile == 64) {
    launch_one_gemm_kernel<256, 64, 64>(grid, a_map, b_map, c_map, d_sink);
""",
            f"""  if (ktiles == {ktiles} && mtile == {mtiles} && ntile == {ntiles}) {{
    launch_one_gemm_kernel<{ktiles}, {mtiles}, {ntiles}>(grid, a_map, b_map, c_map, d_sink);
""",
        )
    text = replace_once(
        text,
        "static constexpr int kPersistentMacroM = 8;",
        f"static constexpr int kPersistentMacroM = {args.macro_m};",
    )
    text = replace_once(
        text,
        "static constexpr int kPersistentMacroN = 16;",
        f"static constexpr int kPersistentMacroN = {args.macro_n};",
    )
    if args.scheduler == "dynamic":
        text = replace_once(
            text,
            "  __shared__ uint32_t warp_sinks[kWarps];\n",
            """  __shared__ uint32_t warp_sinks[kWarps];
  __shared__ int persistent_task_shared;
""",
        )
        text = replace_once(
            text,
            """  int tile_iter = 0;
  int static_linear_tile = static_cast<int>(blockIdx.x);
  while (true) {
    const int linear_tile = static_linear_tile;
    static_linear_tile += static_cast<int>(gridDim.x);
    if (linear_tile >= persistent_task_count)
      break;
""",
            """  int tile_iter = 0;
  while (true) {
    if (threadIdx.x == 0) {
      persistent_task_shared = static_cast<int>(
          atomicAdd(sink + mtile_count * ntile_count, 1u));
    }
    __syncthreads();
    const int linear_tile = persistent_task_shared;
    if (linear_tile >= persistent_task_count)
      break;
""",
        )
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
