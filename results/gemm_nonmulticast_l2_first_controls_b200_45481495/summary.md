# Isolated TMA `evict_first` controls

## Outcome

The single-factor controls reproduce the strong operand asymmetry from the
preceding combined-policy sweep.  Relative to no hint, A-only `evict_first`
reduced throughput by `19.723%`, while B-only `evict_first` reduced it by
`1.642%`.  All three paired samples moved in the same direction for both
controls.

This removes the earlier A/B policy confound: the large A regression does not
require B to be marked `evict_last`.  The result is consistent with the
selected scheduler depending more strongly on retaining A across its longer
reuse pattern.  It still does not measure L2 hit rate or DRAM traffic, and an
eviction priority is a hardware hint rather than guaranteed cache pinning.

## Provenance and fixed conditions

- Definition commit: `1463cea`
- CUDA source SHA-256:
  `8171bbac9aebbe4dce866feb598de16f52ad76275d63d5d303721f098ce689a6`
- Vast instance: `45481495`
- GPU: NVIDIA B200, 148 SMs, 1000 W power limit
- Shape: BF16 `16384x16384x16384`, FP32 accumulation and complete FP32 C
- Input: deterministic BF16 uniform `[0,1)`
- CTA: `256x256`; K64; three SMEM stages; one resident CTA per SM
- Loads per K stage: one `256x64` A TMA and two `64x128` B TMAs
- Scheduler: 148 dynamic persistent CTAs, `16x16` macro, local M-fast and
  macro N-fast
- No multicast and no A/B/C tensor-map L2 promotion
- Effective TMA/MMA phase: `0/0`
- One case per process; warmup 1 and five timed launches
- Three cyclic Latin-order passes; every case occupied each position once

The only binary differences were the A/B eviction policy macros `(0,0)`,
`(2,0)`, and `(0,2)`, where `0=no hint` and `2=L2::evict_first`.  Logical TMA
issue count, requested bytes, coordinates, and C stores were unchanged.

## Performance

All values are event TFLOP/s.  Uncertainty is sample standard deviation across
the three process-level W1/I5 measurements.  Paired change uses the baseline
from the same pass.

| variant | A / B policy | P1 | P2 | P3 | mean TFLOP/s | paired changes | paired mean |
|---|---|---:|---:|---:|---:|---|---:|
| `baseline` | none / none | 1769.789 | 1773.152 | 1769.987 | 1770.976 +/- 1.887 | -- | baseline |
| `a_first` | first / none | 1418.615 | 1415.175 | 1431.259 | 1421.683 +/- 8.470 | -19.843%, -20.189%, -19.137% | **-19.723%** |
| `b_first` | none / first | 1746.456 | 1739.916 | 1739.296 | 1741.889 +/- 3.967 | -1.318%, -1.874%, -1.734% | **-1.642%** |

The magnitudes agree closely with Experiment G's conditional contrasts:
`-19.575%` when A-first was added with B-last fixed and `-1.567%` when
B-first was added with A-last fixed.  Position cannot explain the result,
because each variant ran first, second, and third exactly once.

## Correctness and environment

- All three binaries passed the 512 pattern reference with `max_abs=0`,
  `max_rel=0`, and `bad=0`.
- The host scheduler mapping test reported exact coverage.
- The remote artifact manifest passed `sha256sum -c` before download; all
  downloaded non-binary artifacts passed it again locally.
- GPU temperature was 31 C before and 32 C after; pass-start telemetry was
  31--32 C at an idle-reported 1965 MHz.
- `nvidia-smi` reported no active hardware slowdown or software power cap.
- Pass-start power samples were taken while utilization was zero and are not
  used as workload power evidence.
- Validation covers the 512 pattern test, not a full 16K CPU reference.
- Actual L2 hit rate, L2 bytes, and DRAM bytes remain unmeasured.

## Decision

This closes the single-factor causal diagnostic.  Keep no eviction hint as
the default.  Do not test more `evict_first` combinations or extend them to
other sizes: both operands regress and A regresses severely.  The stable
small `a_last` gain from Experiment G also remains below its selection gate.

## Reproduction

```bash
DEFINITION_COMMIT=1463cea \
  ./run_b200_gemm_nonmulticast_l2_first_controls.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_nonmulticast_l2_first_controls
```

Raw CSVs, validation logs, scheduler mapping output, exact compile commands,
execution sequence, pass-start telemetry, source snapshots, and hashes are
retained in this directory.  Binaries were intentionally omitted from the
download after their hashes were verified remotely.
