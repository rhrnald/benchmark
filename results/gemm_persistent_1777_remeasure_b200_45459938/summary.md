# Frozen-1777 persistent GEMM remeasurement

- Date: 2026-07-21
- Vast.ai instance: `45459938` (destroyed after artifact collection)
- GPU: NVIDIA B200, 148 SM, 1000 W power limit, maximum SM clock 1965 MHz
- Driver/CUDA: 580.126.09 / CUDA 12.9
- Kernel: `gemm256_tma_tcgen05_persistent_1777`
- Inputs: deterministic BF16 uniform `[0,1)`, FP32 accumulation and FP32 TMA C store
- Protocol: one size per process, one warmup launch, five timed launches
- Persistent workers: 148

The 512 validation passed exactly before measurement: `bad=0`, `max_abs=0`,
`max_rel=0`.

| size | event ms | event TFLOP/s | historical 2026-07-20 result | change |
|---:|---:|---:|---:|---:|
| 8192 | 0.649670 | **1692.415** | 1697.631 | -0.31% |
| 16384 | 5.007686 | **1756.518** | 1777.128 | -1.16% |
| 32768 | 43.549324 | **1615.840** | 1615.074 | +0.05% |

The historical result used three warmups and six timed launches per pass on a
different rented B200.  The present result uses the new 1/5 standard, so the
difference cannot be attributed to a kernel change alone.  The 32K result is
effectively reproduced; 16K is 1.16% below the old mean.

Configuration pinned by the build target:

```text
CTA tile                    256x256
K stage                     64
shared-memory stages        3
MMA pipes / issuer warps    2 / 2
persistent workers          148
8K/16K macroblock           16x16
32K macroblock              8x18
local/macro traversal       M-fast / N-fast
A/B TMA L2 promotion        none / none
pipe-1 TMA/MMA phase        0 / 0
```

GPU snapshots were 29 C before the measurements and 33 C immediately after;
these are endpoint snapshots rather than a steady telemetry trace.  Raw CSVs,
console logs, `nvidia-smi -q`, and the measured binary SHA-256 are stored in
this directory.
