#!/usr/bin/env python3
"""Generate the hash-gated, bank-conflict-free vec4 transpose epilogue.

Only the C-staging implementation and its host-visible epilogue label change.
The generated source otherwise remains byte-for-byte identical to the audited
scalar transpose input.

For every four M rows, each contiguous quarter-warp owns one row and each lane
stores four adjacent N values.  A four-way XOR Latin permutation lets four
shuffle instructions transpose the 4x4 register fragment.  Consequently every
quarter-warp STS.128 transaction covers all 32 shared-memory banks exactly once.
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


def make_vec4_cf_stage() -> str:
    values = ", ".join(f"v{i}" for i in range(64))
    steps: list[str] = []
    for quad_m in range(16):
        x0 = quad_m * 4
        x1 = x0 + 1
        x2 = x0 + 2
        x3 = x0 + 3
        swizzle_byte = ((quad_m * 4) & 7) << 4
        steps.extend(
            (
                f'            "xor.b32 rowaddr, lane_swz, '
                f'0x{swizzle_byte:x};\\n\\t"',
                '            "add.u32 rowaddr, rowbase, rowaddr;\\n\\t"',
                # First butterfly: XOR the register index with source-lane j0.
                f'            "selp.b32 a0, v{x1}, v{x0}, lane_j0;\\n\\t"',
                f'            "selp.b32 a1, v{x0}, v{x1}, lane_j0;\\n\\t"',
                f'            "selp.b32 a2, v{x3}, v{x2}, lane_j0;\\n\\t"',
                f'            "selp.b32 a3, v{x2}, v{x3}, lane_j0;\\n\\t"',
                # Second butterfly: XOR with source-lane j1.
                '            "selp.b32 z0, a2, a0, lane_j1;\\n\\t"',
                '            "selp.b32 z1, a3, a1, lane_j1;\\n\\t"',
                '            "selp.b32 z2, a0, a2, lane_j1;\\n\\t"',
                '            "selp.b32 z3, a1, a3, lane_j1;\\n\\t"',
                '            "shfl.sync.idx.b32 u0|valid0, z0, src0, '
                '0x1f, 0xffffffff;\\n\\t"',
                '            "shfl.sync.idx.b32 u1|valid1, z1, src1, '
                '0x1f, 0xffffffff;\\n\\t"',
                '            "shfl.sync.idx.b32 u2|valid2, z2, src2, '
                '0x1f, 0xffffffff;\\n\\t"',
                '            "shfl.sync.idx.b32 u3|valid3, z3, src3, '
                '0x1f, 0xffffffff;\\n\\t"',
                # Undo the same XOR permutation using destination-row q.
                '            "selp.b32 b0, u1, u0, row_q0;\\n\\t"',
                '            "selp.b32 b1, u0, u1, row_q0;\\n\\t"',
                '            "selp.b32 b2, u3, u2, row_q0;\\n\\t"',
                '            "selp.b32 b3, u2, u3, row_q0;\\n\\t"',
                '            "selp.b32 n0, b2, b0, row_q1;\\n\\t"',
                '            "selp.b32 n1, b3, b1, row_q1;\\n\\t"',
                '            "selp.b32 n2, b0, b2, row_q1;\\n\\t"',
                '            "selp.b32 n3, b1, b3, row_q1;\\n\\t"',
                '            "st.shared.v4.b32 [rowaddr], '
                '{n0, n1, n2, n3};\\n\\t"',
            )
        )
        if quad_m != 15:
            steps.append(
                '            "add.u32 rowbase, rowbase, 0x200;\\n\\t"'
            )
    step_text = "\n".join(steps)

    template = r"""// TMEM holds two C-half transposes:
//   pipe 0: rows N[0:128],   columns M[0:256], TMEM columns [0,256)
//   pipe 1: rows N[128:256], columns M[0:256], TMEM columns [256,512)
//
// For every four output M rows, one contiguous quarter-warp owns each row.
// Its eight lanes write all 32 N values as aligned 128-bit stores.  An STS.128
// request is serviced as four quarter-warp transactions, each of which sees a
// permutation of all 32 SW128 banks and is therefore conflict-free.
//
// Source lane 4*g+j owns N component j.  For shuffle phase h it selects M row
// h^j, while destination row q reads source j=q^h.  Thus every shuffle returns
// M row h^(q^h)=q, and an XOR-q output permutation restores N order j=0..3.
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
            ".reg .b32 lane_q, lane_g, lane_j, lane_swz, rowbase, rowaddr, "
            "srcbase, src0, src1, src2, src3, tmp, a0, a1, a2, a3, z0, "
            "z1, z2, z3, u0, u1, u2, u3, b0, b1, b2, b3, n0, n1, n2, "
            "n3;\n\t"
            ".reg .pred lane_j0, lane_j1, row_q0, row_q1, valid0, valid1, "
            "valid2, valid3;\n\t"
            "shr.u32 lane_q, %2, 3;\n\t"
            "and.b32 lane_g, %2, 7;\n\t"
            "and.b32 lane_j, %2, 3;\n\t"
            "and.b32 tmp, lane_j, 1;\n\t"
            "setp.ne.u32 lane_j0, tmp, 0;\n\t"
            "and.b32 tmp, lane_j, 2;\n\t"
            "setp.ne.u32 lane_j1, tmp, 0;\n\t"
            "and.b32 tmp, lane_q, 1;\n\t"
            "setp.ne.u32 row_q0, tmp, 0;\n\t"
            "and.b32 tmp, lane_q, 2;\n\t"
            "setp.ne.u32 row_q1, tmp, 0;\n\t"
            "shl.b32 srcbase, lane_g, 2;\n\t"
            "add.u32 src0, srcbase, lane_q;\n\t"
            "xor.b32 tmp, lane_q, 1;\n\t"
            "add.u32 src1, srcbase, tmp;\n\t"
            "xor.b32 tmp, lane_q, 2;\n\t"
            "add.u32 src2, srcbase, tmp;\n\t"
            "xor.b32 tmp, lane_q, 3;\n\t"
            "add.u32 src3, srcbase, tmp;\n\t"
            "xor.b32 tmp, lane_g, lane_q;\n\t"
            "shl.b32 lane_swz, tmp, 4;\n\t"
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


VEC4_CF_STAGE = make_vec4_cf_stage()

SCALAR_BANNER = '"mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
VEC4_CF_BANNER = '"mma=m128n256k16 epilogue=vec4_cf_shuffle_transpose "'


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
            for quad_m in range(16):
                for quarter in range(4):
                    banks: list[int] = []
                    for lane in range(quarter * 8, quarter * 8 + 8):
                        row_q = lane >> 3
                        n_quad = lane & 7

                        # Symbolically execute all four Latin shuffle phases.
                        u_source_j: list[int] = []
                        u_source_row: list[int] = []
                        for h in range(4):
                            source_j = row_q ^ h
                            source_lane = n_quad * 4 + source_j
                            source_row = h ^ (source_lane & 3)
                            u_source_j.append(source_j)
                            u_source_row.append(source_row)
                        for j in range(4):
                            h = row_q ^ j
                            if u_source_j[h] != j:
                                raise RuntimeError(
                                    "vec4 shuffle selected wrong N source"
                                )
                            if u_source_row[h] != row_q:
                                raise RuntimeError(
                                    "vec4 shuffle selected wrong M row"
                                )

                        local_m = load * 64 + quad_m * 4 + row_q
                        local_n = n_band + n_quad * 4
                        for j in range(4):
                            coordinate = (local_m, local_n + j)
                            if coordinate in coordinates:
                                raise RuntimeError(
                                    f"duplicate output coordinate {coordinate}"
                                )
                            coordinates.add(coordinate)

                        word_offsets = [
                            cstore_word_offset(local_m, local_n + j)
                            for j in range(4)
                        ]
                        if word_offsets[0] & 3:
                            raise RuntimeError("unaligned uint4 word offset")
                        if word_offsets != list(
                            range(word_offsets[0], word_offsets[0] + 4)
                        ):
                            raise RuntimeError("SW128 broke vec4 adjacency")

                        asm_byte_offset = (
                            row_q * 128
                            + quad_m * 512
                            + (
                                (n_quad << 4)
                                ^ (row_q << 4)
                                ^ (((quad_m * 4) & 7) << 4)
                            )
                        )
                        asm_word_offset = (
                            warp_id * 128 * 32
                            + load * 64 * 32
                            + asm_byte_offset // 4
                        )
                        if asm_word_offset != word_offsets[0]:
                            raise RuntimeError(
                                "inline-PTX address does not match SW128 helper"
                            )

                        offsets.update(word_offsets)
                        banks.extend(offset & 31 for offset in word_offsets)

                    if len(set(banks)) != 32:
                        raise RuntimeError(
                            "quarter-warp STS.128 transaction has a bank conflict"
                        )
                    transaction_count += 1

    expected = {(m, n) for m in range(128) for n in range(128)}
    if coordinates != expected:
        raise RuntimeError("vec4 mapping does not cover one 128x128 chunk")
    if len(offsets) != 128 * 128:
        raise RuntimeError("SW128 offsets are not one-to-one")

    return (
        "mapping_audit=pass coordinates=16384 unique_offsets=16384 "
        f"quarterwarp_transactions={transaction_count} "
        "banks_per_transaction=32 conflict_way=1"
    )


def generate(source: str) -> str:
    generated = replace_once(
        source, SCALAR_STAGE, VEC4_CF_STAGE, "scalar transpose C stage"
    )
    generated = replace_once(
        generated, SCALAR_BANNER, VEC4_CF_BANNER, "epilogue banner"
    )

    required = (
        "#pragma unroll 1\n    for (int load = 0;",
        "shfl.sync.idx.b32 u0|valid0, z0, src0",
        "shfl.sync.idx.b32 u3|valid3, z3, src3",
        "st.shared.v4.b32 [rowaddr], {n0, n1, n2, n3}",
        "tcgen05.ld.sync.aligned.32x32b.x64.b32",
        "epilogue=vec4_cf_shuffle_transpose",
    )
    for fragment in required:
        if fragment not in generated:
            raise RuntimeError(f"generated audit: missing {fragment!r}")
    if SCALAR_STAGE in generated or SCALAR_BANNER in generated:
        raise RuntimeError("generated audit: scalar implementation remains")

    reconstructed = replace_once(
        generated, VEC4_CF_STAGE, SCALAR_STAGE, "vec4 CF transpose C stage"
    )
    reconstructed = replace_once(
        reconstructed, VEC4_CF_BANNER, SCALAR_BANNER, "vec4 CF epilogue banner"
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
        f"variant=nsplit_transpose_vec4_cf input_sha256={input_sha256} "
        f"output_sha256={output_sha256} output={args.output}"
    )


if __name__ == "__main__":
    main()
