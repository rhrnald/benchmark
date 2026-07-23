# Current E7a GEMM `clock64` SVG trace

- Date: 2026-07-23
- Vast.ai instance: `45481495` (stopped immediately after download)
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Workload: `16384 x 16384 x 16384`, row-major BF16 A/B, FP32 C
- Input: deterministic BF16 uniform `[0,1)`
- Trace source: exact current-E7a instrumentation source, SHA-256
  `d742bc0331e5faa3b6bbebe4686cdd7222ab97c9a6a416322f69da54c5f4645d`
- Corresponding clean E7a source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a`
- Trace source definition / SVG renderer commits: `87b908b` / `1772148`
- Sampling scope: block 0 on SM 142, ninth valid persistent output tile
  (`tile_iter=8`), real output tile `(28,10)`, linear task 1196
- Protocol: one complete untraced warmup launch followed by exactly one
  traced launch

The trace binary passed size-512 pattern and ones full-C validation exactly:
both reported `max_abs=0`, `max_rel=0`, and `bad=0`.

## Artifact

Open [`gemm_e7a_clock64_trace.svg`](gemm_e7a_clock64_trace.svg).  The upper
panel is the sampled output tile's four-warp timeline.  The lower panel shows
the sums of individually timed consumer calls over all 256 K64 stages.

Only `kind=interval` records are placed on the timeline.  Aggregate wait and
MMA-issue rows overlap in time, so they are drawn as independent bars and must
not be stacked or added into a projected speedup.

## Fresh capture

All values are B200 device `clock64` cycles in one CTA/SM.

| interval or aggregate | W0/W2 | W1/W3 |
|---|---:|---:|
| producer reuse-wait + TMA-issue loop | 264526 | 264492 |
| consumer mainloop + final drain | 267261 | 267359 |
| A-ready wait total, 256 calls | 38692 | 37748 |
| B0-ready wait total, 256 calls | 26700 | 26695 |
| first two wide-MMA issue total, 512 calls | 45056 | 45056 |
| B1-ready wait total, 256 calls | 22214 | 22257 |
| last two wide-MMA issue total, 512 calls | 40704 | 40704 |
| MMA commit total, 256 calls | 17152 | 17152 |
| final MMA drain | 713 | 562 |

Tile-level intervals:

| interval | cycles | complete-tile share |
|---|---:|---:|
| scheduler atomic/barrier/decode envelope | 1179 | 0.422% |
| mainloop start through CTA join | 267702 | 95.80% |
| complete FP32 C epilogue | 10077 | 3.61% |
| complete output tile envelope | **279450** | 100% |

The last producer returned 2914 cycles before the last consumer.  W2 and W3
finished 106 cycles apart.  The fresh complete-tile result differs by only
52 cycles (0.019%) from the earlier five-process `[0,1)` median of 279398
cycles; mainloop-to-join differs by -281 cycles (-0.105%), and the epilogue is
exactly the same 10077 cycles.  The new capture is therefore consistent with
the preserved reference trace.

The scheduler envelope is more variable because the sampled task is acquired
through the global persistent-work atomic.  Even in this capture it is only
0.42% of the tile.

## Interpretation limits

- CUDA exposes this counter as `clock64()`; there is no repository function
  named `cyc64()`.
- These timestamps are comparable only within the sampled CTA on one SM.
- TMA producer bars end when the issuing warp returns from its loop.  They do
  not mark global-memory transaction completion.
- MMA bars measure descriptor setup and asynchronous issue latency, not
  isolated tensor-core execution.  Completion is reflected through buffer
  reuse dependencies and the final drain.
- The trace version uses 186 registers and 3280 B static shared memory versus
  172 registers and 1184 B in the clean kernel.  Clock reads and trace stores
  perturb execution; never report the traced launch as GEMM TFLOP/s.
- This SVG is an output-tile phase trace with per-phase aggregates.  It does
  not expose each individual K64 iteration.  A separate fixed-window
  per-K-stage trace should be used if the next question is exactly which of
  A/B0/B1 readiness gates stalls on each ring-buffer iteration.

## Reproduction

Build and validate:

```bash
/usr/local/cuda-12.9/bin/nvcc -std=c++17 -O3 \
  -gencode arch=compute_100a,code=sm_100a -lineinfo -Xptxas=-v \
  source/dual_wide_phase_trace.cu \
  -o gemm_e7a_clock64_trace -lcuda

./gemm_e7a_clock64_trace --validate --validate-size 512 \
  --validate-pattern pattern
./gemm_e7a_clock64_trace --validate --validate-size 512 \
  --validate-pattern ones
./gemm_e7a_clock64_trace --input-init random \
  --trace-csv csv/trace_random_fresh.csv
```

Render:

```bash
python3 ../../5.GEMM/plot_gemm_e7a_phase_trace.py \
  csv/trace_random_fresh.csv \
  --svg gemm_e7a_clock64_trace.svg \
  --summary-csv aggregate_summary.csv
```
