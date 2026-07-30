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

## Phase 1A — C-store evict-first

The only change is the cache policy of the existing FP32 C TMA store.
A/B loads, output tile order, arithmetic, TMA-store byte count and store wait
sequence are unchanged.  Both binaries use 164 registers with no
stack/local/spill, and both full-C validation patterns pass bit-exactly.

Each cell is four independent W1/I5 processes.  ABBA/BAAB order gives both
variants two first and two second positions.

| input | baseline TFLOP/s | C `evict_first` TFLOP/s | paired change | paired 95% CI |
|---|---:|---:|---:|---:|
| `[0,1)` | 1833.422 | 1831.824 | -0.0870% | [-0.4138%, +0.2397%] |
| `[-8,8)` | 1622.508 | 1622.931 | +0.0261% | [-0.1668%, +0.2191%] |

The effect is statistically neutral and far below the `+0.5%` adoption gate.
The C-store hint is rejected and will not be included in winner combinations.
This also means that C streaming-write cache pollution by itself does not
explain the current cuBLAS gap.

Artifact:
[`gemm_nsplit_l2_phase_4dd4c4c_phase1a`](../results/gemm_nsplit_l2_phase_4dd4c4c_phase1a/)

## Phase 1B — wave-preserving persistent ownership

All variants preserve the exact set of up to 148 logical output tiles in each
wave.  Only the task-to-CTA assignment inside a wave changes.  The
`table_identity` control separates the constant-memory task-table lookup from
the actual ownership permutation.

Offline scheduler metrics:

| variant | same-A transitions | same-B transitions | mean Manhattan distance |
|---|---:|---:|---:|
| direct / table identity | 0 | 0 | 30.599 |
| `wave_a` | 2860 | 0 | 24.258 |
| `wave_b` | 2782 | 52 | 24.222 |
| `wave_balanced` | 2825 | 52 | 24.246 |

There are 3948 per-CTA transitions.  Every mapping is a permutation of all
4096 output tiles, preserves the baseline 27/28-tile CTA load balance and
preserves each wave's unique A/B panel counts.  All generated kernels pass
both 512 validation patterns and compile to 162 registers without
stack/local/spill.

Each performance cell is five independent W1/I5 processes in a 5x5 Latin
order, so each variant occupies every execution position exactly once.

| variant | `[0,1)` TFLOP/s | paired vs direct | `[-8,8)` TFLOP/s | paired vs direct |
|---|---:|---:|---:|---:|
| direct | 1833.406 | reference | 1622.780 | reference |
| `table_identity` | 1824.633 | -0.478% [-0.677,-0.280] | 1621.205 | -0.096% [-0.587,+0.395] |
| `wave_a` | 1821.603 | -0.644% [-0.772,-0.515] | 1613.665 | -0.561% [-0.960,-0.162] |
| `wave_b` | 1804.659 | -1.568% [-1.720,-1.416] | 1590.501 | -1.988% [-2.513,-1.463] |
| `wave_balanced` | 1806.498 | -1.468% [-1.652,-1.283] | 1597.699 | -1.545% [-1.964,-1.125] |

The lookup itself costs up to about 0.48%, but it does not explain the whole
regression of the stronger B/balanced permutations.  Increasing immediate
per-SM panel reuse while preserving the wave footprint is not sufficient and
is actively harmful here.  A likely interpretation is that the original
CTA-to-task relationship interacts better with physical SM/L2 slices and
cross-CTA phase, but hardware counters are unavailable to distinguish these
effects.

All table/permutation candidates are rejected.  Formula-encoding them would
remove some lookup cost but cannot recover the observed 1.5--2.0% B/balanced
regression.  A broad serpentine sweep is therefore not promoted from the
offline gate.

Artifact:
[`gemm_nsplit_l2_phase_4dd4c4c_phase1b`](../results/gemm_nsplit_l2_phase_4dd4c4c_phase1b/)

## Phase 2 — structural B0 phase shift

The clock64 trace showed a 221--524-cycle interval after pipe 0 completion
while warp 0 was still waiting for pipe 1.  Two variants tested whether that
interval can be used:

```text
baseline:
  wait p0 -> wait p1 -> issue A -> issue B0

issue_b0_first:
  wait p0 -> wait p1 -> issue B0 -> issue A

early_b0:
  wait p0 -> issue B0 -> wait p1 -> issue A
```

All variants pass both 512 full-C validation patterns.  Baseline and
`issue_b0_first` use 164 registers; `early_b0` uses 166.  None has
stack/local/spill and tcgen05 already limits residency to one CTA per SM.

Each cell is six independent W1/I5 processes in a balanced six-order design;
each variant occupies each execution position twice.

| variant | `[0,1)` TFLOP/s | paired vs baseline | `[-8,8)` TFLOP/s | paired vs baseline |
|---|---:|---:|---:|---:|
| baseline | 1833.118 | reference | 1621.291 | reference |
| `issue_b0_first` | 1819.487 | -0.743% [-0.887,-0.600] | 1614.013 | -0.449% [-0.622,-0.275] |
| `early_b0` | 1814.207 | -1.032% [-1.171,-0.892] | 1607.838 | -0.830% [-1.025,-0.634] |

The available dependency window is real, but filling it with B0 TMA makes the
whole kernel slower.  Even the issue-order-only control regresses, so this is
not explained solely by the two extra registers in `early_b0`.  The likely
cause is that an earlier B0 request increases TMA/memory-system contention or
causes a less favorable A/B arrival order for the consumers.  Both phase
variants are rejected.

The original Vast host became unavailable while the artifact was being
downloaded.  The six-pass values above were recovered from the live run
stdout; the local artifact currently contains only the raw files transferred
before the connection closed.  The remaining remote files should be fetched
if instance `45481495` becomes available again.

Partial artifact:
[`gemm_nsplit_l2_phase_4dd4c4c_phase2`](../results/gemm_nsplit_l2_phase_4dd4c4c_phase2/)

## Pending

- Phase 3: explicit 3-stage ring
- winner combination and 8K/32K extension

Phase 3 is defined by commit `08e5d62`.  Measurement is waiting for B200
access: the existing Vast instance could not be restarted and creation of a
replacement B200 was rejected with `account lacks credit`.
