# N-split cross-tile first-stage prefetch

Last updated: 2026-07-24

## Goal

This experiment tests whether the next output tile's first K64 input stage can
be hidden under the current tile's FP32 TMA-store epilogue. It keeps the
requested operand ownership:

- one shared A `256x64` BF16 TMA load;
- independent B0/B1 `64x128` BF16 TMA loads;
- warp 2/3 each own one logical `256x128` C half;
- transpose compute uses one `m128n256k16` MMA per consumer and K16.

The parent is the audited scalar-x64 transpose source with SHA-256
`a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc`.
Direct exact remains canonical. Scalar x64 is used here because its faster
mainloop and exposed scalar epilogue make it the relevant overlap candidate.

## Rotating shared-memory hole

One mainloop stage and one C-store buffer are both exactly 64 KiB:

| payload | bytes |
|---|---:|
| A `256x64` BF16 | 32768 |
| B0 `64x128` BF16 | 16384 |
| B1 `64x128` BF16 | 16384 |
| one complete input stage | 65536 |
| one `128x128` FP32 C buffer | 65536 |

The three normal mainloop stages remain physically and logically unchanged:

```text
epoch       = tile_iter * ktiles + kt
stage       = epoch % 3
ready_phase = (epoch / 3) & 1
reuse_phase = ((epoch - 3) / 3) & 1
```

For the next valid output tile:

```text
next_epoch = (tile_iter + 1) * ktiles
reserved   = next_epoch % 3
```

The epilogue stages its two simultaneous C chunks in the other two physical
stages:

| reserved next-input stage | C buffer stages |
|---:|---|
| 0 | 1, 2 |
| 1 | 0, 2 |
| 2 | 0, 1 |

This is preferable to a fixed stage-2 prefetch slot. A fixed slot would collide
with the next tile's normal kt1 or kt2 for two of the three tile epochs and
would shrink the pipeline startup depth. With the rotating hole, prefetched
kt0 uses its normal physical stage and normal barriers; kt1 and kt2 use the
other stages, and kt3 reuses the prefetched stage through the existing
`mma_done` wait.

At 16K, `ktiles=256`, so the reserved stage rotates `1, 2, 0`. At validation
size 512, `ktiles=8` and four persistent output tiles exercise reserved stages
`2, 1, 0`.

## Task and barrier sequence

The first task of each persistent CTA retains the exposed kt0 prologue. During
each valid tile, warp 0 claims the next valid task as soon as it has issued the
current tile's producer loop. The consumers can still be draining the
mainloop tail at that point. Their existing final CTA synchronization publishes
the claimed task to every warp before the epilogue, so no new synchronization
is added just for the atomic. Padded task IDs are skipped without advancing
`tile_iter`; a terminal task never issues TMA.

Before issuing a prefetched kt0, the producers execute the same stage-reuse
waits that the normal kt0 producer would have executed:

- warp 0 waits for both `mma_done[0][reserved]` and
  `mma_done[1][reserved]`, then issues A and B0;
- warp 1 waits for `mma_done[1][reserved]`, then issues B1;
- all three loads use the existing `a_ready[reserved]` and
  `b_ready[*][reserved]` barriers and the normal next-epoch phase.

The next tile's producers start at kt1, while both consumers still start at
kt0. No prefetch-completion wait is added to the transition. The consumers'
existing `a_ready`/`b_ready` waits provide the dependency, and their normal
kt0 commits protect stage reuse at kt3.

## Three-way ablation

| label | definition |
|---|---|
| A: `nsplit_transpose_scalar_x64` | unchanged scalar-x64 parent |
| B: `nsplit_prefetch_nonoverlap` | early task claim, rotating C hole, and split kt0 issue after the complete current epilogue |
| C: `nsplit_prefetch_overlap` | identical to B, except kt0 is issued after the first two C stores are committed and before their `wait_group 0` |

For C, the intended ordering is:

```text
stage and issue C chunks 0 and 1
TMA-store commit_group
next A/B0/B1 reuse waits and TMA issue
TMA-store wait_group 0
existing CTA synchronization
stage and issue C chunks 2 and 3
```

No extra CTA synchronization is inserted around the prefetch. The input loads
use mbarrier completion, while the C stores use the TMA-store bulk group, and
their shared-memory ranges are disjoint. Both B and C execute the same two
noinline helper calls, one at each possible placement. B enables only the late
call and C enables only the early call. Apart from those two uniform Boolean
arguments and the host-visible label, their generated sources are identical.

At dense 16K with 148 persistent CTAs:

- output tiles: 4096;
- exposed CTA prologues: 148;
- cross-tile kt0 prefetches: 3948;
- terminal prefetches: 0;
- task-counter atomics: `4096 + 148 = 4244`;
- per output tile: 768 input TMA loads, 2048 MMA instructions, and four C
  TMA stores, unchanged from scalar x64.

## Correctness and local gates

Before a B200 performance run:

1. Hash-generate B and C from the exact scalar-x64 parent.
2. Prove the rotating C buffers are distinct, 64 KiB aligned, and disjoint
   from the reserved stage for ktile counts 4, 8, and 256.
3. Prove every valid kt0 is issued exactly once, either in the first-tile
   prologue or in the preceding transition.
4. Confirm mainloop stage indices and ready/reuse phases are unchanged.
5. Require `kStages==3`, `kCStoreBuffers==2`, and
   `kCStoreStageWords==kStageWords` with static assertions.
6. Build for `sm_100a`; require zero stack, local memory, and spills.
7. Require B and C to have identical registers and shared-memory resources,
   an identical opcode-plus-modifier SASS multiset, and identical
   MMA/TMEM-load/shared-store/TMA-store counts. Their ordered SASS must differ,
   proving that the active issue was actually moved.
8. Run full-C `pattern256`, `pattern512`, and `ones512` validation for A, B,
   and C. The 512 cases are mandatory because they cross three tile
   transitions and exercise every reserved physical stage.

The final local `sm_100a` gate emits 2,832 kernel instructions, 194 registers,
1,184 B static shared memory, and zero stack/local/spill for both candidates.
Each has 4 static MMA sites, 24 input-TMA sites, 4 output-TMA sites, 54 phase
checks, 8 x64 TMEM-load sites, 512 scalar shared-store sites, and 5 noinline
calls.

The performance binary contains no trace or debug counters.

## B200 screening protocol

For each of BF16 uniform `[0,1)` and `[-8,8)`, run all six permutations of
A/B/C. Every cell is a separate process with one warmup and five timed
launches. This gives 36 processes total, two observations in every position,
and two occurrences of every directed non-self adjacency.

Report pass-paired process-level percentage changes with a Student-t 95%
confidence interval, df=5:

- primary: C versus B, the isolated overlap-placement effect;
- net: C versus A;
- diagnostic: B versus A, the early-claim/rotating-buffer/split-kt0 cost.

C advances only if both inputs have a strictly positive C/B confidence-interval
lower bound and a nonnegative C/A lower bound. Since A is not the canonical
direct-exact source, a passing screen is not sufficient for adoption. A
separately committed same-session confirmation must compare C with direct
exact and require at least `+0.5%` paired mean improvement for both inputs with
a positive confidence-interval lower bound.

No statistical outlier is deleted. An infrastructure failure invalidates and
reruns the complete affected ordered triplet.

## Instance workflow

All source editing, local generation, codegen checks, documentation, and the
definition commit are completed while the Vast instance is stopped. Only
then:

1. activate the instance;
2. clone the exact committed definition;
3. validate and run the B200 experiment;
4. create and download the full checksummed artifact;
5. verify the local archive and internal manifest;
6. stop the instance immediately;
7. analyze, curate, document, and commit the result locally.

Local code-generation gate:

```bash
out_dir=$(mktemp -d /tmp/gemm-prefetch-codegen.XXXXXX)
GEMM_CODEGEN_ONLY=1 \
  ./run_b200_gemm_nsplit_cross_tile_prefetch.sh \
  "$(pwd)" "$out_dir/result"
```

The committed B200 runner performs the same code-generation gate, nine full-C
validations, and then the 36 independent W1/I5 processes:

```bash
./run_b200_gemm_nsplit_cross_tile_prefetch.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_cross_tile_prefetch_b200
```
