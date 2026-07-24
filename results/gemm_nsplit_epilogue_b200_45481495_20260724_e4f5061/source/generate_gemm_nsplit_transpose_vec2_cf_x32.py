#!/usr/bin/env python3
"""Generate a bank-conflict-free x32 vec2 transpose epilogue.

This candidate reuses the audited x32 TMEM-load stage, but assigns one
half-warp to each row of a 2x2 output fragment.  Each 16-lane, 128-byte
shared-memory transaction then covers all 32 banks exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from generate_gemm_nsplit_transpose_vec2_x32 import (
    EXPECTED_INPUT_SHA256,
    SCALAR_BANNER,
    SCALAR_STAGE,
    VEC2_X32_STAGE,
    replace_once,
)


INTERLEAVED_MAPPING = r"""    const int lane_in_pair = lane & 1;
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
"""

BANK_CF_MAPPING = r"""    const int row_in_pair = lane >> 4;
    const int n_pair = lane & 15;
    const int local_n_pair = n_band + n_pair * 2;
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
        const bool source_n_odd = (lane & 1) != 0;
        const uint32_t z_a = source_n_odd ? odd_m : even_m;
        const uint32_t z_b = source_n_odd ? even_m : odd_m;
        const int even_n_lane = n_pair * 2;
        const int odd_n_lane = even_n_lane + 1;
        const uint32_t u = __shfl_sync(
            0xffffffffu, z_a,
            row_in_pair ? odd_n_lane : even_n_lane);
        const uint32_t v = __shfl_sync(
            0xffffffffu, z_b,
            row_in_pair ? even_n_lane : odd_n_lane);
        const uint32_t n0 = row_in_pair ? v : u;
        const uint32_t n1 = row_in_pair ? u : v;
        const int local_m = load * 32 + pair_m * 2 + row_in_pair;
        const int word_offset =
            cstore_sw128_float_word_offset(local_m, local_n_pair);
        reinterpret_cast<uint2 *>(c_smem + word_offset)[0] =
            make_uint2(n0, n1);
      }
    }
"""

INTERLEAVED_COMMENT = r"""// Lane pair (2g, 2g+1) owns adjacent output columns (2g, 2g+1).
// For each adjacent M-row pair it exchanges one register in each direction,
// so the even and odd lanes respectively hold the two values for the even
// and odd M rows.  Each lane then emits one aligned 64-bit shared store.
//
// The SW128 row XOR is a multiple of four words.  It cannot change bit zero
// of the even logical column, so the physical vec2 address remains 8-byte
// aligned and its two FP32 words remain adjacent.
"""

BANK_CF_COMMENT = r"""// Lanes 0..15 emit the even M row and lanes 16..31 emit the odd M
// row of each 2x2 fragment.  Two indexed shuffles gather adjacent N values.
// Consequently each half-warp emits one 128-byte transaction whose 32 words
// cover all shared-memory banks exactly once.
//
// The SW128 row XOR is a multiple of four words.  It cannot change bit zero
// of the even logical column, so the physical vec2 address remains 8-byte
// aligned and its two FP32 words remain adjacent.
"""

INTERLEAVED_BANNER = (
    '"mma=m128n256k16 epilogue=vec2_x32_shuffle_transpose "'
)
BANK_CF_BANNER = (
    '"mma=m128n256k16 epilogue=vec2_cf_x32_shuffle_transpose "'
)


def make_bank_cf_stage() -> str:
    stage = replace_once(
        VEC2_X32_STAGE,
        INTERLEAVED_COMMENT,
        BANK_CF_COMMENT,
        "x32 mapping comment",
    )
    return replace_once(
        stage,
        INTERLEAVED_MAPPING,
        BANK_CF_MAPPING,
        "x32 lane mapping",
    )


def generate(source: str) -> str:
    bank_cf_stage = make_bank_cf_stage()
    generated = replace_once(
        source, SCALAR_STAGE, bank_cf_stage, "scalar transpose C stage"
    )
    generated = replace_once(
        generated, SCALAR_BANNER, BANK_CF_BANNER, "epilogue banner"
    )

    required = (
        "tcgen05.ld.sync.aligned.32x32b.x32.b32",
        "const int row_in_pair = lane >> 4;",
        "const int n_pair = lane & 15;",
        "const uint32_t u = __shfl_sync(",
        "const uint32_t v = __shfl_sync(",
        "reinterpret_cast<uint2 *>(c_smem + word_offset)[0]",
        "epilogue=vec2_cf_x32_shuffle_transpose",
    )
    for fragment in required:
        if fragment not in generated:
            raise RuntimeError(f"generated audit: missing {fragment!r}")
    if INTERLEAVED_MAPPING in generated or INTERLEAVED_BANNER in generated:
        raise RuntimeError(
            "generated audit: interleaved x32 implementation remains"
        )

    reconstructed = replace_once(
        generated, bank_cf_stage, SCALAR_STAGE, "bank-CF x32 C stage"
    )
    reconstructed = replace_once(
        reconstructed, BANK_CF_BANNER, SCALAR_BANNER, "bank-CF x32 banner"
    )
    if reconstructed != source:
        raise RuntimeError(
            "generated audit: a region outside the stage or banner changed"
        )
    return generated


def cstore_word_offset(row: int, col: int) -> int:
    """Host equivalent of cstore_sw128_float_word_offset."""
    return (
        (col >> 5) * (128 * 32)
        + row * 32
        + ((col & 31) ^ ((row & 7) << 2))
    )


def audit_host_mapping() -> tuple[int, int]:
    """Exhaustively prove the shuffle mapping, coverage, and bank pattern."""
    writes: dict[int, tuple[int, int]] = {}
    halfwarp_transactions = 0

    for warp in range(4):
        n_band = warp * 32
        for load in range(4):
            for pair_m in range(16):
                even_row = load * 32 + pair_m * 2
                odd_row = even_row + 1
                even = [(even_row, n_band + lane) for lane in range(32)]
                odd = [(odd_row, n_band + lane) for lane in range(32)]
                z_a = [
                    odd[lane] if lane & 1 else even[lane]
                    for lane in range(32)
                ]
                z_b = [
                    even[lane] if lane & 1 else odd[lane]
                    for lane in range(32)
                ]

                lane_words: list[tuple[int, int]] = []
                for lane in range(32):
                    row_in_pair = lane >> 4
                    n_pair = lane & 15
                    even_n_lane = n_pair * 2
                    odd_n_lane = even_n_lane + 1
                    u = z_a[
                        odd_n_lane if row_in_pair else even_n_lane
                    ]
                    v = z_b[
                        even_n_lane if row_in_pair else odd_n_lane
                    ]
                    n0, n1 = (v, u) if row_in_pair else (u, v)
                    row = even_row + row_in_pair
                    col = n_band + even_n_lane
                    expected = ((row, col), (row, col + 1))
                    if (n0, n1) != expected:
                        raise RuntimeError(
                            "host mapping audit: shuffle mismatch "
                            f"lane={lane} got={(n0, n1)} expected={expected}"
                        )

                    dst = cstore_word_offset(row, col)
                    if dst & 1 or cstore_word_offset(row, col + 1) != dst + 1:
                        raise RuntimeError(
                            "host mapping audit: vec2 is not aligned/adjacent"
                        )
                    for word, value in ((dst, n0), (dst + 1, n1)):
                        if word in writes:
                            raise RuntimeError(
                                f"host mapping audit: duplicate word {word}"
                            )
                        writes[word] = value
                    lane_words.append((dst, dst + 1))

                for halfwarp in (lane_words[:16], lane_words[16:]):
                    banks = [0] * 32
                    for vec2 in halfwarp:
                        for word in vec2:
                            banks[word & 31] += 1
                    if banks != [1] * 32:
                        raise RuntimeError(
                            "host mapping audit: half-warp bank conflict "
                            f"counts={banks}"
                        )
                    halfwarp_transactions += 1

    expected_words = 128 * 128
    if len(writes) != expected_words:
        raise RuntimeError(
            "host mapping audit: incomplete coverage "
            f"got={len(writes)} expected={expected_words}"
        )
    for row in range(128):
        for col in range(128):
            word = cstore_word_offset(row, col)
            if writes.get(word) != (row, col):
                raise RuntimeError(
                    "host mapping audit: final permutation mismatch "
                    f"row={row} col={col}"
                )
    return expected_words, halfwarp_transactions


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

    words, transactions = audit_host_mapping()
    print(
        f"mapping_audit=exact words={words} aligned_vec2=all "
        f"halfwarp_transactions={transactions} bank_multiplicity=1"
    )
    generated = generate(input_bytes.decode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    output_sha256 = hashlib.sha256(generated.encode("utf-8")).hexdigest()
    print(
        f"variant=nsplit_transpose_vec2_cf_x32 input_sha256={input_sha256} "
        f"output_sha256={output_sha256} output={args.output}"
    )


if __name__ == "__main__":
    main()
