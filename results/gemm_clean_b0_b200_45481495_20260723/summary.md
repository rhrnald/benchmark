# Clean 16K GEMM B0 equivalence gate

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Definition commit: `8add902`
- Clean source SHA-256:
  `6a084088e121b3c87a54aface12b9a7cc060acbde148d267c33aeb6558af0c0e`
- Clean binary SHA-256:
  `c8cf7f415a8933c516d90725f46eceb14e66fc0168f0fffe45566d048e904667`
- Preserved `p0` SHA-256:
  `166044d6690b52dc448befefb79baabb1c9cb56950b16a0536454249d623ff72`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-rotated process pairs per input distribution

The instance was activated only after the definition commit.  The GPU was
32 C before validation and 35 C after all performance runs.  The observed SM
clock before every timed process except the first cold query was 1965 MHz.

## Correctness and codegen

| binary | validation input | max abs | max rel | result |
|---|---|---:|---:|---|
| clean | pattern | 0 | 0 | pass |
| clean | ones | 0 | 0 | pass |
| p0 | pattern | 0 | 0 | pass |
| p0 | ones | 0 | 0 | pass |

The clean SM100a kernel compiled with 178 registers, a 16-byte stack, no
spills, 1184 B static shared memory reported by cuobjdump, and 197632 B dynamic
shared memory.  These values match the selected kernel in `p0`.

## Performance

All values below are event TFLOP/s.  `delta` is clean relative to `p0`; paired
delta is the mean of the three same-pass ratios.

| input | clean samples | clean mean | p0 samples | p0 mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1739.412, 1738.739, 1736.445 | **1738.199** | 1738.912, 1740.249, 1738.169 | **1739.110** | -0.0524% | -0.0524% |
| BF16 uniform `[-8,8)` | 1509.497, 1508.256, 1509.185 | **1508.979** | 1516.698, 1509.716, 1510.746 | **1512.387** | -0.2253% | -0.2249% |

Both differences are below the predefined 0.5% noise threshold.  B0 therefore
passes: removing tuning macros, unused runtime modes, trace ABI, and dead
specializations did not materially change performance.

The current-instance `p0` `[0,1)` result is 3.7388% below its historical
1806.657 TFLOP/s result.  The clean binary is not the cause because the
preserved `p0` moved by the same amount in the same session.  The historical
run used driver 595.71.05 on instance 45465499, whereas this run used driver
580.126.09 on instance 45481495.  This establishes an environment difference,
but the available data do not isolate driver versus host/GPU variation.
Temperature is not a plausible explanation here because the current run was
colder.

The performance-path `checksum=0` is only a scheduler diagnostic.  Correctness
is established by the full 512 C comparison above.

Decision: use the clean kernel from `8add902` as the source baseline for the
next one-factor, non-L2 hot-path ablations.
