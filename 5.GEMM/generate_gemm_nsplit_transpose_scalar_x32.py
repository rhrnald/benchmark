#!/usr/bin/env python3
"""Generate a scalar x32-TMEM-load transpose epilogue.

Only the TMEM load granularity, C-staging implementation, and host-visible
epilogue label change. The input is hash-gated to the audited scalar
transpose source, and reconstruction proves that no other source region was
modified.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_INPUT_SHA256 = (
    "a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc"
)

SCALAR_X64_STAGE = r"""// TMEM holds two C-half transposes:
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

SCALAR_X32_STAGE = r"""// Load one TMEM row's next 32 FP32 columns per lane.
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
// This is the scalar-coalesced mapping used by the x64 control. Each warp
// owns one N32 band, and each store instruction writes one conflict-free
// 128-byte output-row segment. Only the live TMEM fragment changes from
// 64 to 32 registers.
__device__ __forceinline__ void
stage_float_c_chunk(uint32_t tmem_base, uint32_t *c_smem, int chunk_m,
                    int chunk_n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (warp_id < kCStoreWarps) {
    uint32_t r[32];
    const int n_band = warp_id * 32;
    const uint32_t pipe_tmem =
        tmem_base + chunk_n * kTransposeTmemPipeStride;
#pragma unroll 1
    for (int load = 0; load < kCStoreChunkM / 32; ++load) {
      const uint32_t global_m_base =
          static_cast<uint32_t>(chunk_m * kCStoreChunkM + load * 32);
      const uint32_t row_taddr =
          pipe_tmem + (static_cast<uint32_t>(n_band) << 16) + global_m_base;
      tcgen05_ld_32x32b_x32(r, row_taddr);
      tcgen05_wait_ld();
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        const int local_m = load * 32 + i;
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

SCALAR_X64_BANNER = '"mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
SCALAR_X32_BANNER = (
    '"mma=m128n256k16 epilogue=scalar_x32_coalesced_transpose "'
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
    """Prove exact coverage and minimum-bank-wavefront scalar stores."""
    coordinates: set[tuple[int, int]] = set()
    tmem_sources: set[tuple[int, int]] = set()
    warp_store_transactions = 0

    for chunk_m in range(2):
        for chunk_n in range(2):
            chunk_offsets: set[int] = set()
            for warp_id in range(4):
                n_band = warp_id * 32
                for load in range(4):
                    for i in range(32):
                        banks: set[int] = set()
                        for lane in range(32):
                            local_m = load * 32 + i
                            local_n = n_band + lane
                            coordinate = (
                                chunk_m * 128 + local_m,
                                chunk_n * 128 + local_n,
                            )
                            source = (
                                n_band + lane,
                                chunk_n * 256
                                + chunk_m * 128
                                + load * 32
                                + i,
                            )
                            offset = cstore_word_offset(local_m, local_n)
                            if (
                                coordinate in coordinates
                                or source in tmem_sources
                                or offset in chunk_offsets
                            ):
                                raise RuntimeError(
                                    "duplicate scalar x32 mapping at "
                                    f"output={coordinate} source={source}"
                                )
                            coordinates.add(coordinate)
                            tmem_sources.add(source)
                            chunk_offsets.add(offset)
                            banks.add(offset & 31)
                        if len(banks) != 32:
                            raise RuntimeError(
                                "scalar x32 store has a bank conflict"
                            )
                        warp_store_transactions += 1
            if len(chunk_offsets) != 128 * 128:
                raise RuntimeError("chunk SW128 mapping is not one-to-one")

    expected_coordinates = {
        (m, n) for m in range(256) for n in range(256)
    }
    expected_sources = {
        (row, col) for row in range(128) for col in range(512)
    }
    if (
        coordinates != expected_coordinates
        or tmem_sources != expected_sources
    ):
        raise RuntimeError("full scalar x32 mapping is not one-to-one")
    return (
        "mapping_audit=exact words=65536 tmem_words=65536 "
        f"warp_store_transactions={warp_store_transactions} "
        "unique_banks_per_transaction=32"
    )


def generate(source: str) -> str:
    generated = replace_once(
        source,
        SCALAR_X64_STAGE,
        SCALAR_X32_STAGE,
        "scalar x64 transpose C stage",
    )
    generated = replace_once(
        generated,
        SCALAR_X64_BANNER,
        SCALAR_X32_BANNER,
        "epilogue banner",
    )

    required = (
        "tcgen05.ld.sync.aligned.32x32b.x32.b32",
        "uint32_t r[32];",
        "#pragma unroll 1\n"
        "    for (int load = 0; load < kCStoreChunkM / 32; ++load)",
        "c_smem[cstore_sw128_float_word_offset(local_m, local_n)] = r[i]",
        "epilogue=scalar_x32_coalesced_transpose",
    )
    for fragment in required:
        if fragment not in generated:
            raise RuntimeError(f"generated audit: missing {fragment!r}")
    if "__shfl" in SCALAR_X32_STAGE:
        raise RuntimeError("scalar x32 stage unexpectedly contains a shuffle")
    if SCALAR_X64_STAGE in generated or SCALAR_X64_BANNER in generated:
        raise RuntimeError("generated audit: scalar x64 implementation remains")
    if generated.count("tcgen05_ld_32x32b_x32(") != 2:
        raise RuntimeError(
            "generated audit: expected one x32 helper and one call site"
        )

    reconstructed = replace_once(
        generated,
        SCALAR_X32_STAGE,
        SCALAR_X64_STAGE,
        "scalar x32 C stage",
    )
    reconstructed = replace_once(
        reconstructed,
        SCALAR_X32_BANNER,
        SCALAR_X64_BANNER,
        "scalar x32 banner",
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
        f"variant=nsplit_transpose_scalar_x32 "
        f"input_sha256={input_sha256} output_sha256={output_sha256} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
