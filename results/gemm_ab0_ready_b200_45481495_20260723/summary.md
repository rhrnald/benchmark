# E2c combined A and B0 readiness barrier

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Macro-free E2a parent: `3d2d0a4`
- E2c definition: `77e8900`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Baseline source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- Candidate source SHA-256:
  `6a59331ac87128667a8381e918b59fb81967c79879584413d9f1e153aa52601f`
- Baseline / candidate binary SHA-256:
  `5d4127162b83ec0584b760290d4f7542b71d7588838a012fa4201e74cd7f6023` /
  `bf3f49d0f6044517bbedbe87b368fb5e0276d581a1e3d5575caec97ac9f6cd65`

Warp 0's 32 KiB A and 16 KiB B0 TMA loads use one 48 KiB transaction
barrier.  Consumer pipe 0 therefore performs one ready wait rather than two;
pipe 1 waits for the combined A+B0 barrier and its independent B1 barrier.
TMA bytes, load instructions, MMA work, stage count, and tile order are
unchanged.

Candidate pattern and ones full-C validations at size 512 passed exactly.

## Code generation

| metric | macro-free E2a | combined A+B0 barrier |
|---|---:|---:|
| registers / spills | 174 / 0 | 174 / 0 |
| stack | 0 B | 0 B |
| static shared | 1184 B | 1152 B |
| `SYNCS.ARRIVE` | 21 | 14 |
| TMA load / MMA / TMA store / CTA barriers | 21 / 8 / 4 / 20 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass parent.

| input | parent samples | parent mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1748.783, 1749.277, 1747.811 | **1748.624** | 1738.301, 1738.536, 1736.783 | **1737.873** | **-0.6148%** | **-0.6148%** |
| BF16 uniform `[-8,8)` | 1518.478, 1516.645, 1519.265 | **1518.129** | 1498.265, 1487.291, 1490.370 | **1491.975** | **-1.7228%** | **-1.7228%** |

All six paired deltas are negative.  The combined barrier adds B0 to pipe 1's
dependency set even though pipe 1 only consumes A and B1.  The reduced barrier
instruction count does not compensate for that head-of-line dependency, with
a particularly large penalty for `[-8,8)`.  E2c is rejected without an
extended run; the independent readiness barriers must be preserved.
