# Non-multicast L2 scheduler round 1

## Outcome

At `M=N=K=16384`, changing both scheduler levels from local M-fast / macro
N-fast (`order_mn`) to local N-fast / macro M-fast (`order_nm`) improved the
three-process mean from `1769.017` to `1778.907 TFLOP/s`, a paired mean gain
of `0.559%`.  All three paired samples were positive (`+0.400%`, `+0.747%`,
and `+0.530%`).  This is a candidate, not yet the default, because the gain is
close to the plan's 0.5% selection threshold and requires a focused rerun.

Macro-row snake traversal was effectively neutral (`+0.085%` to `+0.136%`).
Dynamic N-strip ownership regressed (`-0.238%` for S=2 and `-1.222%` for
S=4), so completing an entire K=16384 output before revisiting the same A
coordinates did not create useful enough L2 reuse.

## Fixed conditions

- Definition commit: `f669b6f`
- Vast instance: `45481495`
- GPU: NVIDIA B200, 148 SMs, 1000 W power limit
- Inputs: deterministic BF16 uniform `[0,1)`
- Output: complete FP32 C through the TMA store path
- Shape: `16384 x 16384 x 16384`
- CTA: `256x256`; K stage 64; 3 shared-memory stages
- A TMA per stage: one `256x64` BF16 tile
- B TMA per stage: two `64x128` BF16 tiles
- Persistent dynamic scheduler: 148 CTAs, no multicast
- Timing: one process per case, warmup 1, five timed launches
- Sampling: three forward/reverse/rotated process passes
- Effective TMA/MMA phase: `0/0`
- TMA L2 promotion: disabled for A, B, and C

## Performance

The uncertainty is the sample standard deviation across three independent
processes.  Delta is the mean of the three pass-wise percentage changes from
`order_mn`.

| variant | local order | macro order / change | P1 | P2 | P3 | mean TFLOP/s | event ms | delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `order_mn` | M-fast | N-fast | 1770.546 | 1768.668 | 1767.838 | 1769.017 +/- 1.387 | 4.972308 +/- 0.003898 | baseline |
| `order_mm` | M-fast | M-fast | 1740.758 | 1740.110 | 1741.773 | 1740.880 +/- 0.838 | 5.052670 +/- 0.002432 | -1.590% |
| `order_nn` | N-fast | N-fast | 1757.509 | 1757.475 | 1755.518 | 1756.834 +/- 1.140 | 5.006788 +/- 0.003250 | -0.689% |
| `order_nm` | N-fast | M-fast | 1777.629 | 1781.881 | 1777.210 | **1778.907 +/- 2.584** | **4.944670 +/- 0.007177** | **+0.559%** |
| `snake_1` | M-fast | reverse macro N on odd rows | 1770.867 | 1770.137 | 1770.555 | 1770.520 +/- 0.366 | 4.968086 +/- 0.001028 | +0.085% |
| `snake_2` | M-fast | reverse macro and local N | 1772.002 | 1770.678 | 1771.605 | 1771.428 +/- 0.679 | 4.965538 +/- 0.001904 | +0.136% |
| `strip_2` | M-fast | one owner computes 2 N tiles | 1765.621 | 1765.789 | 1762.994 | 1764.801 +/- 1.567 | 4.984186 +/- 0.004429 | -0.238% |
| `strip_4` | M-fast | one owner computes 4 N tiles | 1744.559 | 1748.251 | 1749.408 | 1747.406 +/- 2.533 | 5.033807 +/- 0.007301 | -1.222% |

`order_nm` versus `order_mn` pass-wise deltas were `+0.400%`, `+0.747%`,
and `+0.530%`.

## Correctness and environment

- All eight binaries passed the 512 pattern validation with `max_abs=0`,
  `max_rel=0`, and `bad=0`.
- The host scheduler test covered baseline, snake, and strip mappings and
  reported exact, unique coverage.
- GPU temperature was 31 C before and 33 C after the sweep.  No thermal
  slowdown or hardware slowdown was reported by `nvidia-smi`.
- Nsight Compute counters are unavailable on this rented host.  Snake and
  strip variants request the same logical A/B payload as the baseline, but
  actual L2 and DRAM bytes were not measured.

## Interpretation and next gate

The winning candidate places same-A N neighbors next to each other inside a
macro while retaining the same B-coordinate range across adjacent M macros.
This is consistent with A locality being the larger opportunity in the prior
partial-reuse experiment, but throughput alone does not prove that cache
traffic fell.

Run a focused `order_mn` versus `order_nm` AB/BA confirmation before changing
the compile-time default.  Reject snake and S=2/S=4 strip ownership for this
kernel.  If the order result confirms, retune only a small macro-shape set
under `order_nm`; if it does not, retain `order_mn` and move to the planned
K-outer direct-A-reuse design audit.

## Artifacts

- `csv/`: 24 raw one-row CSV files
- `validate_*.log`: per-binary numerical validation
- `mapping_test.log`: host scheduler coverage test
- `raw.log`: complete benchmark stdout
- `source/`: exact source, plan, test, and run-script snapshot
- `sha256.txt`, `nvcc_version.txt`, `nvidia_smi_before.txt`,
  `nvidia_smi_after.txt`: provenance and environment

Reproduce on a B200 from the repository root with:

```bash
DEFINITION_COMMIT=f669b6f ./run_b200_gemm_nonmulticast_l2_round1.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_nonmulticast_l2_round1
```
