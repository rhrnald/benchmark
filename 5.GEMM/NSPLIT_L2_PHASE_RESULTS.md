# N-split L2 / structural phase results

Last updated: 2026-07-30

Definition plan:
[`NSPLIT_L2_PHASE_OPTIMIZATION_PLAN.md`](NSPLIT_L2_PHASE_OPTIMIZATION_PLAN.md)

## Phase 0 — current-pipeline clock64 trace

Definition commit: `4dd4c4c`

The trace variant samples block 0's ninth persistent output tile
(`linear_tile=1184`, output tile `(16,20)`) over K64 stages 56--66.  Eight
core stages include all producer/consumer events and three context stages
close the producer's 3-stage reuse observations.

Both `pattern` and `ones` full-C 512 validation passed bit-exactly for the
baseline, trace and C-store-hint binaries.  The uninstrumented baseline and
C-store-hint performance instances both compile to 164 registers with zero
stack/local/spill.

Producer wait summary:

| event | samples | mean cycles | median | min | max |
|---|---:|---:|---:|---:|---:|
| warp 0 wait pipe 0 | 11 | 416.4 | 396 | 254 | 553 |
| warp 0 additional wait pipe 1 | 11 | 320.5 | 373 | 221 | 524 |
| warp 1 wait pipe 1 | 11 | 870.9 | 864 | 741 | 1014 |

Every sampled stage had a nonzero second wait after warp 0 had already
observed pipe 0 completion.  The additional pipe-1 wait was 221--524 cycles.
This directly supports testing:

```text
wait pipe0 -> issue B0 -> wait pipe1 -> issue A
```

instead of waiting for both pipes before issuing B0.  The trace does not prove
a throughput gain: instrumentation changes one CTA and a TMA issued in the gap
can contend with other traffic.  It establishes that the dependency split has
a real overlap window rather than relying on an arbitrary sleep.

Other diagnostic means over the eight core stages:

| event | mean cycles |
|---|---:|
| A TMA prepare/issue | 92.1 |
| B0 TMA prepare/issue | 78.0 |
| B1 TMA prepare/issue | 100.2 |
| consumer 0 MMA issue span | 525.6 |
| consumer 1 MMA issue span | 616.5 |
| full epilogue | 12664 |

Artifact:
[`gemm_nsplit_l2_phase_4dd4c4c_phase0`](../results/gemm_nsplit_l2_phase_4dd4c4c_phase0/)

## Pending

- Phase 1A: C-store `evict_first`
- Phase 1B: wave-preserving persistent ownership
- Phase 2: B0/A issue order and early-B0 dependency split
- Phase 3: explicit 3-stage ring
- winner combination and 8K/32K extension
