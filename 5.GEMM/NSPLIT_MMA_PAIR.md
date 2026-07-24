# N-split paired ordinary-MMA codegen gate

## Question

Can the requested N-split mainloop reduce operand-descriptor lowering
overhead by placing the two same-B `m128n128k16` operations in one inline PTX
block?

The exact N-split issues, for every K16:

```text
MMA A[top M128]    x B[N pipe]
MMA A[bottom M128] x B[same N pipe]
```

The candidate preserves the compact shared runtime-pipe consumer, all
barriers, TMA transactions, TMEM addresses, accumulation predicates, commit,
scheduler, and epilogue.  It removes only the unrolled two-iteration
M-block C++ loop and calls one inline helper containing the same two ordinary
`tcgen05.mma` PTX instructions.  The consumer outer K64 loop carries
`#pragma unroll 1`; the previously measured control proved that this pragma
produces byte-identical normalized SASS for the exact ordinary-MMA source,
and it prevents the paired helper from triggering an unrelated three-stage
auto-unroll.

## Predeclared codegen gate

The candidate is compiled locally for SM100a before any B200 activation.  It
advances to correctness and timing only if normalized main-kernel SASS:

- still contains eight static ordinary `UTCHMMA` sites, corresponding to 16
  dynamic MMA instructions per CTA and K64 across the two consumer warps;
- does not duplicate the runtime-pipe consumer body;
- reduces repeated operand-lowering instructions or otherwise shortens the
  hot consumer sequence without adding spill, stack, or local memory.

If generated SASS is equivalent or larger with no reduction around each MMA
pair, this direction is rejected without spending a remote GPU run.

The generator is hash-gated to exact N-split SHA-256
`cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`.

## Codegen result

The gate ran from definition commit
`8dca23bef0941554deb7e3c500075afe32a85f91` with local CUDA 12.9.  The
generated source SHA-256 is
`1fe1b9987066c9a3df1c8b1f4720f9126187da8912dd29b37c4407da30ef5063`.

| normalized main-kernel metric | exact N-split | paired helper |
|---|---:|---:|
| SASS instruction lines | 1913 | 1917 |
| registers / stack / spill | 174 / 0 / 0 | 174 / 0 / 0 |
| static ordinary `UTCHMMA` | 8 | 8 |
| `ELECT` | 32 | 32 |
| `R2UR.BROADCAST` | 65 | 65 |
| branch sites | 154 | 154 |
| `BSSY` / `BSYNC` | 18 / 18 | 18 / 18 |
| `SYNCS.PHASECHK` | 48 | 48 |

Putting two PTX instructions in one inline-assembly block does not make ptxas
retain the common B descriptor in uniform registers: every MMA still has its
own `ELECT` and descriptor broadcasts.  The candidate instead adds four
normalized instructions.  It therefore fails the predeclared SASS gate and
is rejected without activating the B200 instance.

Generated source, build/resource output, normalized SASS, metrics, and
hashes are retained in
[`../results/gemm_nsplit_mma_pair_codegen_sm100a_20260724_8dca23b/`](../results/gemm_nsplit_mma_pair_codegen_sm100a_20260724_8dca23b/).
