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
