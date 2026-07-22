# Non-multicast TMA L2 eviction-priority sweep

## Outcome

No eviction-priority variant passed the predeclared promotion gate.  Keeping
the A loads at `L2::evict_last` was faster than the no-hint baseline in all
three paired passes, but the mean gain was only `+0.113%`, well below the
required `+0.5%`.  The compile-time default therefore remains no cache hint.

The sweep did reveal a strong asymmetry.  Marking A `evict_first` while B was
`evict_last` reduced throughput by `19.552%`; the reverse assignment reduced
it by only `1.456%`.  With B fixed at `evict_last`, adding A `evict_first`
cost `19.575%` relative to `b_last`; with A fixed at `evict_last`, adding B
`evict_first` cost `1.567%` relative to `a_last`.  This is consistent with A
residency being more important under the selected local-M-fast / macro-N-fast
schedule.  It is not the unconditional effect of either `evict_first` hint,
nor a direct measurement of L2 hit rate or DRAM bytes.  PTX policies are hints
and performance-counter access was unavailable.

## Provenance and fixed conditions

- Definition commit: `1bb691b`
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
- Three forward/reverse/rotated process passes

Policy encoding is `0=no hint`, `1=L2::evict_last`, and
`2=L2::evict_first`.  Policies were applied only to input TMA loads; C stores
were unchanged.  Logical TMA issue count and requested payload were identical
across all variants.

## Performance

All values below are event TFLOP/s.  Uncertainty is sample standard deviation
across the three process-level W1/I5 measurements.  Paired change is computed
against the baseline from the same pass.

| variant | A / B policy | P1 | P2 | P3 | mean TFLOP/s | paired changes | paired mean |
|---|---|---:|---:|---:|---:|---|---:|
| `baseline` | none / none | 1770.696 | 1768.695 | 1768.199 | 1769.197 +/- 1.322 | -- | baseline |
| `a_last` | last / none | 1772.969 | 1770.286 | 1770.320 | **1771.192 +/- 1.539** | +0.128%, +0.090%, +0.120% | **+0.113%** |
| `b_last` | none / last | 1768.666 | 1771.034 | 1769.406 | 1769.702 +/- 1.211 | -0.115%, +0.132%, +0.068% | +0.029% |
| `a_last_b_first` | last / first | 1742.376 | 1742.827 | 1745.113 | 1743.439 +/- 1.467 | -1.599%, -1.463%, -1.306% | -1.456% |
| `a_first_b_last` | first / last | 1415.324 | 1427.952 | 1426.585 | 1423.287 +/- 6.930 | -20.070%, -19.265%, -19.320% | -19.552% |

`a_last` satisfies the three-for-three direction condition but fails the
minimum-effect condition.  No focused confirmation or 8K/32K extension is
warranted by the declared gate.

## Correctness and environment

- All five binaries passed the 512 pattern reference with `max_abs=0`,
  `max_rel=0`, and `bad=0`.
- The host scheduler mapping test reported exact coverage.
- The remote artifact manifest passed `sha256sum -c` before download.
- GPU temperature was 31 C before and 32 C after; pass-start telemetry was
  31--32 C at an idle-reported 1965 MHz.
- `nvidia-smi` reported no active hardware slowdown or software power cap.
- Actual L2 hit rate, L2 bytes, and DRAM bytes remain unmeasured.

## Decision

Keep no TMA L2 eviction hint as the default.  Do not spend GPU time on a
focused `a_last` confirmation or a size sweep: its stable `+0.113%` effect is
too small to meet the `+0.5%` threshold.  Treat the asymmetric negative
controls as evidence conditional on the opposite operand's `evict_last`
setting.  A separate A-first-only/B-first-only sweep is needed before making
an unconditional per-operand claim; any locality interpretation remains a
throughput-based inference rather than a cache-counter result.

## Reproduction

```bash
DEFINITION_COMMIT=1bb691b \
  ./run_b200_gemm_nonmulticast_l2_evict.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_nonmulticast_l2_evict
```

Raw CSVs, validation logs, scheduler mapping output, exact compile commands,
execution sequence, pass-start telemetry, source snapshots, and hashes are
retained in this directory.  Binaries were intentionally omitted from the
download after their hashes were verified remotely.
