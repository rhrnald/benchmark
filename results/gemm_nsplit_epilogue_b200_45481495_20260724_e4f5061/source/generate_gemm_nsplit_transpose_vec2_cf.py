#!/usr/bin/env python3
"""Generate the hash-gated, bank-conflict-free vec2 transpose epilogue.

Only the C-staging implementation and its host-visible epilogue label change.
The generated source otherwise remains byte-for-byte identical to the audited
scalar transpose input.

The scalar source leaves one logical N column in each lane.  For every two M
rows, this implementation makes lanes 0..15 own the even row and lanes 16..31
own the odd row.  Each half-warp therefore writes one complete 128-byte SMEM
row with sixteen aligned 64-bit stores.  Two shuffles gather the adjacent N
pair without mixing the two rows in either half-warp transaction.
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

def make_vec2_cf_stage() -> str:
    values = ", ".join(f"v{i}" for i in range(64))
    steps: list[str] = []
    for pair_m in range(32):
        even = pair_m * 2
        odd = even + 1
        swizzle_byte = ((pair_m * 2) & 7) << 4
        steps.extend(
            (
                f'            "xor.b32 rowaddr, lane_swz, '
                f'0x{swizzle_byte:x};\\n\\t"',
                '            "add.u32 rowaddr, rowbase, rowaddr;\\n\\t"',
                f'            "selp.b32 z_a, v{odd}, v{even}, '
                f'lane_odd;\\n\\t"',
                f'            "selp.b32 z_b, v{even}, v{odd}, '
                f'lane_odd;\\n\\t"',
                '            "shfl.sync.idx.b32 u|valid_a, z_a, src_a, '
                '0x1f, 0xffffffff;\\n\\t"',
                '            "shfl.sync.idx.b32 v|valid_b, z_b, src_b, '
                '0x1f, 0xffffffff;\\n\\t"',
                '            "selp.b32 n0, v, u, row_odd;\\n\\t"',
                '            "selp.b32 n1, u, v, row_odd;\\n\\t"',
                '            "st.shared.v2.b32 [rowaddr], {n0, n1};\\n\\t"',
            )
        )
        if pair_m != 31:
            steps.append(
                '            "add.u32 rowbase, rowbase, 0x100;\\n\\t"'
            )
    step_text = "\n".join(steps)

    template = r"""// TMEM holds two C-half transposes:
//   pipe 0: rows N[0:128],   columns M[0:256], TMEM columns [0,256)
//   pipe 1: rows N[128:256], columns M[0:256], TMEM columns [256,512)
//
// For every two output M rows, lanes 0..15 own the even row and lanes 16..31
// own the odd row.  Each half-warp writes all 32 N values of exactly one row
// as sixteen aligned 64-bit stores.  This is important: an all-lane STS.64
// request is serviced as two half-warp transactions, and interleaving the two
// rows by lane parity would make their SW128 bank-pair sets collide two-way.
//
// Source lanes 2*g and 2*g+1 hold adjacent N values.  z_a/z_b form a Latin
// permutation of the two M-row registers, so two shuffles deliver both values
// to either destination half-warp without a third lane-routing shuffle.
__device__ __forceinline__ void
stage_float_c_chunk(uint32_t tmem_base, uint32_t *c_smem, int chunk_m,
                    int chunk_n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (warp_id < kCStoreWarps) {
    const int n_band = warp_id * 32;
    const uint32_t pipe_tmem =
        tmem_base + chunk_n * kTransposeTmemPipeStride;
#pragma unroll 1
    for (int load = 0; load < kCStoreChunkM / 64; ++load) {
      const uint32_t global_m_base =
          static_cast<uint32_t>(chunk_m * kCStoreChunkM + load * 64);
      const uint32_t row_taddr =
          pipe_tmem + (static_cast<uint32_t>(n_band) << 16) + global_m_base;
      const uint32_t dst_base = smem_ptr_u32(
          c_smem + warp_id * kCStoreChunkM * 32 + load * 64 * 32);
      asm volatile(
            "{\n\t"
            ".reg .b32 @VALUES@;\n\t"
            ".reg .b32 lane_q, lane_pair, lane_col, lane_swz, rowbase, "
            "rowaddr, src_even, src_a, src_b, tmp, z_a, z_b, u, v, n0, "
            "n1;\n\t"
            ".reg .pred lane_odd, row_odd, valid_a, valid_b;\n\t"
            "shr.u32 lane_q, %2, 4;\n\t"
            "and.b32 lane_pair, %2, 0xf;\n\t"
            "and.b32 tmp, %2, 1;\n\t"
            "setp.ne.u32 lane_odd, tmp, 0;\n\t"
            "setp.ne.u32 row_odd, lane_q, 0;\n\t"
            "shl.b32 src_even, lane_pair, 1;\n\t"
            "add.u32 src_a, src_even, lane_q;\n\t"
            "xor.b32 tmp, lane_q, 1;\n\t"
            "add.u32 src_b, src_even, tmp;\n\t"
            "shl.b32 lane_col, lane_pair, 3;\n\t"
            "shl.b32 tmp, lane_q, 4;\n\t"
            "xor.b32 lane_swz, lane_col, tmp;\n\t"
            "shl.b32 tmp, lane_q, 7;\n\t"
            "add.u32 rowbase, %1, tmp;\n\t"
            "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
            "{@VALUES@}, [%0];\n\t"
            "tcgen05.wait::ld.sync.aligned;\n\t"
@STEPS@
            "}"
            :
            : "r"(row_taddr), "r"(dst_base), "r"(lane)
            : "memory");
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
    return template.replace("@VALUES@", values).replace("@STEPS@", step_text)


VEC2_CF_STAGE = make_vec2_cf_stage()

SCALAR_BANNER = '"mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
VEC2_CF_BANNER = '"mma=m128n256k16 epilogue=vec2_cf_shuffle_transpose "'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def cstore_word_offset(row: int, col: int) -> int:
    """Host copy of cstore_sw128_float_word_offset."""
    col_block = col >> 5
    in_block = col & 31
    return col_block * (128 * 32) + row * 32 + (
        in_block ^ ((row & 7) << 2)
    )


def audit_mapping() -> str:
    """Exhaustively verify one complete 128x128 C-staging chunk."""
    coordinates: set[tuple[int, int]] = set()
    offsets: set[int] = set()
    transaction_count = 0

    for warp_id in range(4):
        n_band = warp_id * 32
        for load in range(2):
            for pair_m in range(32):
                for half in range(2):
                    banks: list[int] = []
                    for lane in range(half * 16, half * 16 + 16):
                        row_in_pair = lane >> 4
                        n_pair = lane & 15
                        source_even_n = n_pair * 2
                        source_odd_n = source_even_n + 1

                        # Symbolically execute the two-shuffle Latin mapping.
                        source_a = (
                            source_odd_n if row_in_pair else source_even_n
                        )
                        source_b = (
                            source_even_n if row_in_pair else source_odd_n
                        )
                        # z_a is odd_m in odd source lanes, even_m otherwise;
                        # z_b is the complementary row.
                        u_row = source_a & 1
                        v_row = 1 - (source_b & 1)
                        n0_source_row = v_row if row_in_pair else u_row
                        n1_source_row = u_row if row_in_pair else v_row
                        if n0_source_row != row_in_pair:
                            raise RuntimeError("n0 shuffle selected wrong M row")
                        if n1_source_row != row_in_pair:
                            raise RuntimeError("n1 shuffle selected wrong M row")

                        local_m = load * 64 + pair_m * 2 + row_in_pair
                        local_n = n_band + n_pair * 2
                        expected_sources = (source_even_n, source_odd_n)
                        for j, source_lane in enumerate(expected_sources):
                            if source_lane != n_pair * 2 + j:
                                raise RuntimeError("wrong source N lane")
                            coordinate = (local_m, local_n + j)
                            if coordinate in coordinates:
                                raise RuntimeError(
                                    f"duplicate output coordinate {coordinate}"
                                )
                            coordinates.add(coordinate)

                        word_offset = cstore_word_offset(local_m, local_n)
                        next_offset = cstore_word_offset(local_m, local_n + 1)
                        if word_offset & 1:
                            raise RuntimeError("unaligned uint2 word offset")
                        if next_offset != word_offset + 1:
                            raise RuntimeError("SW128 broke vec2 adjacency")
                        asm_byte_offset = (
                            row_in_pair * 128
                            + pair_m * 256
                            + (
                                (n_pair * 8)
                                ^ (row_in_pair * 16)
                                ^ (((pair_m * 2) & 7) << 4)
                            )
                        )
                        asm_word_offset = (
                            warp_id * 128 * 32
                            + load * 64 * 32
                            + asm_byte_offset // 4
                        )
                        if asm_word_offset != word_offset:
                            raise RuntimeError(
                                "inline-PTX address does not match SW128 helper"
                            )
                        offsets.update((word_offset, next_offset))
                        banks.extend((word_offset & 31, next_offset & 31))

                    if len(set(banks)) != 32:
                        raise RuntimeError(
                            "half-warp STS.64 transaction has a bank conflict"
                        )
                    transaction_count += 1

    expected = {(m, n) for m in range(128) for n in range(128)}
    if coordinates != expected:
        raise RuntimeError("vec2 mapping does not cover one 128x128 chunk")
    if len(offsets) != 128 * 128:
        raise RuntimeError("SW128 offsets are not one-to-one")

    return (
        "mapping_audit=pass coordinates=16384 unique_offsets=16384 "
        f"halfwarp_transactions={transaction_count} "
        "banks_per_transaction=32 conflict_way=1"
    )


def generate(source: str) -> str:
    generated = replace_once(
        source, SCALAR_STAGE, VEC2_CF_STAGE, "scalar transpose C stage"
    )
    generated = replace_once(
        generated, SCALAR_BANNER, VEC2_CF_BANNER, "epilogue banner"
    )

    required = (
        "#pragma unroll 1\n    for (int load = 0;",
        "shfl.sync.idx.b32 u|valid_a, z_a, src_a",
        "shfl.sync.idx.b32 v|valid_b, z_b, src_b",
        "st.shared.v2.b32 [rowaddr], {n0, n1}",
        "tcgen05.ld.sync.aligned.32x32b.x64.b32",
        "epilogue=vec2_cf_shuffle_transpose",
    )
    for fragment in required:
        if fragment not in generated:
            raise RuntimeError(f"generated audit: missing {fragment!r}")
    if SCALAR_STAGE in generated or SCALAR_BANNER in generated:
        raise RuntimeError("generated audit: scalar implementation remains")

    reconstructed = replace_once(
        generated, VEC2_CF_STAGE, SCALAR_STAGE, "vec2 CF transpose C stage"
    )
    reconstructed = replace_once(
        reconstructed, VEC2_CF_BANNER, SCALAR_BANNER, "vec2 CF epilogue banner"
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
        f"variant=nsplit_transpose_vec2_cf input_sha256={input_sha256} "
        f"output_sha256={output_sha256} output={args.output}"
    )


if __name__ == "__main__":
    main()
