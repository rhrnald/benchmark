# Historical 1797 repeated-tile GEMM remeasurement

- GPU: NVIDIA B200, Vast instance `45481495`
- Source: exact 2026-07-18 snapshot, hashes in `sha256.txt`
- Work: 592 CTAs, repeated random BF16 `[0,1)` A/B `128x128` panels,
  two independent FP32 `128x128` accumulators, K-panel 128
- Timing: N/2N differential, `steps=8192/16384`
- Standard: one warmup and five timed launches per N/2N run, three processes
- Validation: PASS, `bad=0`, `max_abs=6.10352e-05`

| process | TFLOP/s |
|---:|---:|
| 1 | 1798.802266 |
| 2 | 1807.125275 |
| 3 | 1794.919926 |
| mean ± population stddev | **1800.282489 ± 5.091557** |

The exact historical kernel reproduces the old 1797.227 result under the new
`warmup=1`, `iters=5` standard. The current integrated `128x256, K=128`
baseline's 1712.793 TFLOP/s is therefore a software-structure regression, not
a failure to reproduce the old GPU clock or input-power state.

The main structural difference is operand reuse. This kernel loads one
`128x128` A panel and one `128x128` B panel per step, then both issuer warps
reuse that same B panel for the two N accumulators. Its stage is 64 KiB and it
fits three stages. The integrated kernel loads distinct left/right B panels,
making a stage 96 KiB and limiting the pipeline to two stages.
