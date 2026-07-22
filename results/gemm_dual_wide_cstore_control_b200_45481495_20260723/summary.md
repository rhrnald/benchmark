# Dual-wide same-binary C-store upper bound

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide default: `ef7ebca`
- Runtime-control definition: `5bdf116`
- Protocol: one case per process, warmup 1, timed launches 5, three cyclically
  ordered clean/store-on/store-off triples per input distribution
- Clean / control source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `fdff3665fc44dcf18fff71e901cbbf788bcb70d9923d63b63b672eca4440de18`
- Clean / control binary SHA-256:
  `91ad00034ea7a3c9a8aaf1db76c0e100104c85cf38b61900d828e994ea3b400f` /
  `1e042eaa26952f6f5b0bcb5af754662bad849f24280108ace6f4d0037d7628ad`

The control binary takes one uniform runtime argument.  Store-on executes the
normal TMEM-to-SMEM and TMA C-store path; store-off skips only that path.  The
final MMA completion waits, caller CTA barrier, persistent scheduler, sink
publication, register allocation, and all non-epilogue work are common.

Store-on pattern and ones full-C validations at size 512 passed exactly.
Store-off intentionally does not produce C, so its TFLOP/s is only a nominal
diagnostic upper bound.

## Code generation

| metric | clean default | runtime-control binary |
|---|---:|---:|
| registers / spills | 172 / 0 | 172 / 0 |
| stack / static shared | 0 B / 1184 B | unchanged |
| all SASS instructions | 4192 | 4208 |
| static MMA / A TMA / B TMA | 4 / 7 / 14 | unchanged |
| C TMEM loads / TMA stores | 8 / 4 | unchanged |
| `UTCBAR` / `BAR.SYNC` | 1 / 20 | unchanged |

## Store-on equivalence gate

| input | clean samples | clean mean | control store-on samples | store-on mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1774.150, 1771.806, 1759.088 | **1768.348** | 1771.477, 1773.269, 1778.383 | **1774.376** | **+0.3409%** | **+0.3429%** |
| BF16 uniform `[-8,8)` | 1527.874, 1533.920, 1530.025 | **1530.606** | 1527.458, 1533.792, 1527.677 | **1529.642** | **-0.0630%** | **-0.0630%** |

Both distributions pass the 0.5% same-work equivalence gate.  The third
random clean process is a low outlier, but cyclic ordering and the signed
control keep the runtime branch within the predefined noise gate.

## Strict store-off upper bound

Delta is same-binary store-off relative to same-pass store-on.

| input | store-on samples | store-on mean | store-off samples | store-off mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1771.477, 1773.269, 1778.383 | **1774.376** | 1830.360, 1831.984, 1830.633 | **1830.992** | **+3.1908%** | **+3.1910%** |
| BF16 uniform `[-8,8)` | 1527.458, 1533.792, 1527.677 | **1529.642** | 1579.996, 1581.893, 1578.354 | **1580.081** | **+3.2974%** | **+3.2976%** |

The complete dual-wide C epilogue therefore costs at most about 3.2--3.3% of
throughput.  This remains meaningful, but even eliminating it entirely would
reach only about 1.831 PFLOP/s for `[0,1)`.  The next trace and barrier
ablations should still prioritize mainloop synchronization and issue overhead;
epilogue overlap cannot by itself close the cuBLAS gap.
