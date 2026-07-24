# N-split weight-stationary B-collector experiment

## Question

Can `tcgen05.mma.ws` recover the extra issue cost of the requested N-split
mainloop by reusing each K16xN128 B operand across its two M128 MMAs?

In the exact N-split kernel, each consumer performs this sequence for every
K16:

```text
B descriptor kk
  -> MMA A[M0:128, kk] x B[kk, N-pipe]
  -> MMA A[M128:256, kk] x B[kk, N-pipe]
```

The B descriptor is identical for the pair, but ordinary `tcgen05.mma` does
not request persistent B-collector reuse.  After controlling for consumer
specialization, the candidate changes only the two MMA opcodes:

```text
first M128  : tcgen05.mma.ws ... collector::bN::fill
second M128 : tcgen05.mma.ws ... collector::bN::lastuse
```

Warp 2 owns collector `b0` and warp 3 owns collector `b1`.  The two
independent issuing threads never share a collector.  `lastuse` discards the
collector after the second M128 reads it; the next K16 can then refill that
warp's collector.  The existing `mma_done` completion barrier remains
unchanged, so neither A nor B SMEM is overwritten before every asynchronous
MMA using that stage completes.

The static control and WS candidate use the same `Pipe`-specialized consumer.
Local CUDA 12.9 compilation gives both generated kernels 166 registers with
no spill; the original runtime-pipe N-split uses 174 registers.  Comparing
the WS candidate directly only with the original would therefore confound
collector reuse with compiler specialization.  The static control makes the
two effects separately measurable.  The normalized static-control SASS has
16 ordinary `UTCHMMA` instructions.  The WS SASS has no ordinary `UTCHMMA`
and exactly 16 `UTCHMMA.WS`: eight `B_KEEP`, eight `B_REUSE`, with eight
instructions assigned to `BUFFER1` for warp 3.  Both have 41 `ELECT`
instructions.

The instruction descriptor, A/B SMEM descriptors, TMEM addresses,
accumulation predicate, commit path, TMA transactions, scheduler, and
epilogue are unchanged.  The official PTX ISA defines `fill` as reading B
from memory into collector `bN`, and `lastuse` as reading that collector and
then discarding it:

<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma-ws>

## Variants

| variant | only changed mechanism |
|---|---|
| `e7a_exact` | exact M-split performance reference |
| `nsplit_exact` | exact requested N-split, runtime pipe selection |
| `nsplit_static_control` | same static consumer as WS, ordinary MMA |
| `nsplit_ws_b01` | N-split with W2=`b0`, W3=`b1`, M0 fill/M1 lastuse |

The generator is hash-gated to exact N-split SHA-256
`cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`.

## Protocol

- one NVIDIA B200, 148 persistent CTAs
- dense `M=N=K=16384`, complete real A/B coordinates and FP32 C store
- BF16 uniform `[0,1)` and uniform `[-8,8)` inputs
- one process per case, one warmup and five timed launches
- four-pass Latin rotation; every variant occupies each run position once
- size-512 pattern and ones full-C validation before timing
- SM clock, power, and temperature telemetry immediately before and after
  every timed process
- source/binary hashes, compiler resources, full SASS, and normalized
  main-kernel instruction streams retained
- specialization decision against pass-matched `nsplit_exact`
- collector decision against pass-matched `nsplit_static_control`

The candidate advances only if it improves both input distributions by at
least 0.5% relative to the static control.  Replacing the E7a reference
additionally requires beating `e7a_exact` in both distributions.

## Reproduction

```bash
./run_b200_gemm_nsplit_ws.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_ws_b200
```
