#!/usr/bin/env python3
"""Generate an x32-TMEM-load vec2 epilogue from the scalar transpose source.

Only the C-staging implementation and its host-visible epilogue label change.
The input is hash-gated to the audited scalar transpose artifact, and the
generator proves by reconstruction that no other source region was modified.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_INPUT_SHA256 = (
    "a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc"
)

SCALAR_STAGE = r"""// TMEM holds two C-half transposes:
//   pipe 0: rows N[0:128],   columns M[0:256], TMEM columns [0,256)
//   pipe 1: rows N[128:256], columns M[0:256], TMEM columns [256,512)
//
// One warp owns 32 TMEM rows.  For each output M row, its 32 lanes write 32
// adjacent logical N values.  cstore_sw128_float_word_offset applies only a
// lane permutation within that 128-byte row segment, so every scalar warp
// store is conflict-free and covers one complete 128-byte shared-memory line.
__device__ __forceinline__ void
stage_float_c_chunk(uint32_t tmem_base, uint32_t *c_smem, int chunk_m,
                    int chunk_n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (warp_id < kCStoreWarps) {
    uint32_t r[64];
    const int n_band = warp_id * 32;
    const uint32_t pipe_tmem =
        tmem_base + chunk_n * kTransposeTmemPipeStride;
#pragma unroll
    for (int load = 0; load < kCStoreChunkM / 64; ++load) {
      const uint32_t global_m_base =
          static_cast<uint32_t>(chunk_m * kCStoreChunkM + load * 64);
      const uint32_t row_taddr =
          pipe_tmem + (static_cast<uint32_t>(n_band) << 16) + global_m_base;
      tcgen05_ld_32x32b_x64(r, row_taddr);
      tcgen05_wait_ld();
#pragma unroll
      for (int i = 0; i < 64; ++i) {
        const int local_m = load * 64 + i;
        const int local_n = n_band + lane;
        c_smem[cstore_sw128_float_word_offset(local_m, local_n)] = r[i];
      }
    }
  }
#else
  (void)tmem_base;
  (void)c_smem;
  (void)chunk_m;
  (void)chunk_n;
#endif
}

"""

VEC2_X32_STAGE = r"""// Load one TMEM row's next 32 FP32 columns per lane.
// Keeping this helper local to the generated epilogue leaves the audited
// x64 helper and the rest of the kernel byte-for-byte unchanged.
__device__ __forceinline__ void
tcgen05_ld_32x32b_x32(uint32_t (&dst)[32], uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
      "%28, %29, %30, %31}, [%32];"
      : "=&r"(dst[0]), "=&r"(dst[1]), "=&r"(dst[2]), "=&r"(dst[3]),
        "=&r"(dst[4]), "=&r"(dst[5]), "=&r"(dst[6]), "=&r"(dst[7]),
        "=&r"(dst[8]), "=&r"(dst[9]), "=&r"(dst[10]), "=&r"(dst[11]),
        "=&r"(dst[12]), "=&r"(dst[13]), "=&r"(dst[14]), "=&r"(dst[15]),
        "=&r"(dst[16]), "=&r"(dst[17]), "=&r"(dst[18]), "=&r"(dst[19]),
        "=&r"(dst[20]), "=&r"(dst[21]), "=&r"(dst[22]), "=&r"(dst[23]),
        "=&r"(dst[24]), "=&r"(dst[25]), "=&r"(dst[26]), "=&r"(dst[27]),
        "=&r"(dst[28]), "=&r"(dst[29]), "=&r"(dst[30]), "=&r"(dst[31])
      : "r"(taddr)
      : "memory");
#else
  (void)taddr;
  for (int i = 0; i < 32; ++i)
    dst[i] = 0;
#endif
}

// TMEM holds two C-half transposes:
//   pipe 0: rows N[0:128],   columns M[0:256], TMEM columns [0,256)
//   pipe 1: rows N[128:256], columns M[0:256], TMEM columns [256,512)
//
// Lane pair (2g, 2g+1) owns adjacent output columns (2g, 2g+1).
// For each adjacent M-row pair it exchanges one register in each direction,
// so the even and odd lanes respectively hold the two values for the even
// and odd M rows.  Each lane then emits one aligned 64-bit shared store.
//
// The SW128 row XOR is a multiple of four words.  It cannot change bit zero
// of the even logical column, so the physical vec2 address remains 8-byte
// aligned and its two FP32 words remain adjacent.
__device__ __forceinline__ void
stage_float_c_chunk(uint32_t tmem_base, uint32_t *c_smem, int chunk_m,
                    int chunk_n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (warp_id < kCStoreWarps) {
    uint32_t r[32];
    const int n_band = warp_id * 32;
    const int lane_in_pair = lane & 1;
    const int local_n_pair = n_band + (lane >> 1) * 2;
    const uint32_t pipe_tmem =
        tmem_base + chunk_n * kTransposeTmemPipeStride;
// A runtime four-step loop bounds each live TMEM fragment to 32 registers.
#pragma unroll 1
    for (int load = 0; load < kCStoreChunkM / 32; ++load) {
      const uint32_t global_m_base =
          static_cast<uint32_t>(chunk_m * kCStoreChunkM + load * 32);
      const uint32_t row_taddr =
          pipe_tmem + (static_cast<uint32_t>(n_band) << 16) + global_m_base;
      tcgen05_ld_32x32b_x32(r, row_taddr);
      tcgen05_wait_ld();
#pragma unroll
      for (int pair_m = 0; pair_m < 16; ++pair_m) {
        const uint32_t even_m = r[pair_m * 2];
        const uint32_t odd_m = r[pair_m * 2 + 1];
        const uint32_t peer =
            __shfl_xor_sync(0xffffffffu,
                            lane_in_pair ? even_m : odd_m, 1, 2);
        const uint32_t n0 = lane_in_pair ? peer : even_m;
        const uint32_t n1 = lane_in_pair ? odd_m : peer;
        const int local_m = load * 32 + pair_m * 2 + lane_in_pair;
        const int word_offset =
            cstore_sw128_float_word_offset(local_m, local_n_pair);
        reinterpret_cast<uint2 *>(c_smem + word_offset)[0] =
            make_uint2(n0, n1);
      }
    }
  }
#else
  (void)tmem_base;
  (void)c_smem;
  (void)chunk_m;
  (void)chunk_n;
#endif
}

"""

SCALAR_BANNER = '"mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
VEC2_X32_BANNER = (
    '"mma=m128n256k16 epilogue=vec2_x32_shuffle_transpose "'
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def cstore_word_offset(row: int, col: int) -> int:
    col_block = col >> 5
    in_block = col & 31
    return col_block * (128 * 32) + row * 32 + (
        in_block ^ ((row & 7) << 2)
    )


def audit_mapping() -> str:
    """Verify exact output coverage and record the intentional bank conflict."""
    coordinates: set[tuple[int, int]] = set()
    offsets: set[int] = set()
    transactions = 0

    for warp_id in range(4):
        n_band = warp_id * 32
        for load in range(4):
            for pair_m in range(16):
                for half in range(2):
                    bank_counts = [0] * 32
                    for lane in range(half * 16, half * 16 + 16):
                        row_in_pair = lane & 1
                        n_pair = lane >> 1
                        local_m = load * 32 + pair_m * 2 + row_in_pair
                        local_n = n_band + n_pair * 2
                        word_offset = cstore_word_offset(local_m, local_n)
                        next_offset = cstore_word_offset(local_m, local_n + 1)
                        if word_offset & 1 or next_offset != word_offset + 1:
                            raise RuntimeError("invalid vec2 alignment")
                        for j, offset in enumerate((word_offset, next_offset)):
                            coordinate = (local_m, local_n + j)
                            if coordinate in coordinates:
                                raise RuntimeError(
                                    f"duplicate output coordinate {coordinate}"
                                )
                            coordinates.add(coordinate)
                            offsets.add(offset)
                            bank_counts[offset & 31] += 1
                    if max(bank_counts) != 2:
                        raise RuntimeError(
                            "adjacent mapping no longer has the expected "
                            "two-way half-warp bank conflict"
                        )
                    transactions += 1

    expected = {(m, n) for m in range(128) for n in range(128)}
    if coordinates != expected or len(offsets) != 128 * 128:
        raise RuntimeError("x32 vec2 mapping is not one-to-one")
    return (
        "mapping_audit=exact words=16384 aligned_vec2=all "
        f"halfwarp_transactions={transactions} bank_multiplicity=2"
    )


def generate(source: str) -> str:
    generated = replace_once(
        source, SCALAR_STAGE, VEC2_X32_STAGE, "scalar transpose C stage"
    )
    generated = replace_once(
        generated, SCALAR_BANNER, VEC2_X32_BANNER, "epilogue banner"
    )

    required = (
        "tcgen05.ld.sync.aligned.32x32b.x32.b32",
        "uint32_t r[32];",
        "#pragma unroll 1\n"
        "    for (int load = 0; load < kCStoreChunkM / 32; ++load)",
        "__shfl_xor_sync(0xffffffffu,",
        "reinterpret_cast<uint2 *>(c_smem + word_offset)[0]",
        "epilogue=vec2_x32_shuffle_transpose",
    )
    for fragment in required:
        if fragment not in generated:
            raise RuntimeError(f"generated audit: missing {fragment!r}")
    if SCALAR_STAGE in generated or SCALAR_BANNER in generated:
        raise RuntimeError("generated audit: scalar implementation remains")
    if generated.count("tcgen05_ld_32x32b_x32(") != 2:
        raise RuntimeError(
            "generated audit: expected one x32 helper and one call site"
        )

    reconstructed = replace_once(
        generated, VEC2_X32_STAGE, SCALAR_STAGE, "x32 vec2 C stage"
    )
    reconstructed = replace_once(
        reconstructed, VEC2_X32_BANNER, SCALAR_BANNER, "x32 vec2 banner"
    )
    if reconstructed != source:
        raise RuntimeError(
            "generated audit: a region outside the stage or banner changed"
        )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "--source", dest="input", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            "refusing to patch an unaudited scalar transpose source: "
            f"expected {EXPECTED_INPUT_SHA256}, got {input_sha256}"
        )

    mapping_audit = audit_mapping()
    generated = generate(input_bytes.decode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    output_sha256 = hashlib.sha256(generated.encode("utf-8")).hexdigest()
    print(mapping_audit)
    print(
        f"variant=nsplit_transpose_vec2_x32 input_sha256={input_sha256} "
        f"output_sha256={output_sha256} output={args.output}"
    )


if __name__ == "__main__":
    main()
