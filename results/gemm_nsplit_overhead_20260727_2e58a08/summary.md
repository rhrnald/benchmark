# Static 8x16 overhead ablation

16K square BF16 GEMM, FP32 accumulation/output, 148 persistent CTAs,
static 8x16 M-fast scheduling, W1/I5, four position-balanced process
samples per cell.

## `[0,1)`

| variant | TFLOP/s mean +/- SD | paired delta vs baseline |
|---|---:|---:|
| `baseline` | 1840.266 +/- 2.197 | +0.000% +/- 0.000%p |
| `no_memset` | 1838.057 +/- 1.868 | -0.120% +/- 0.180%p |
| `sink_keep` | 1837.966 +/- 3.082 | -0.125% +/- 0.201%p |
| `sink_trim` | 1838.834 +/- 1.381 | -0.078% +/- 0.174%p |
| `fixed` | 1832.372 +/- 2.655 | -0.429% +/- 0.172%p |
| `suspend` | 1823.834 +/- 1.743 | -0.893% +/- 0.046%p |
| `fixed_sink` | 1833.652 +/- 2.148 | -0.359% +/- 0.225%p |
| `all` | 1823.715 +/- 2.490 | -0.899% +/- 0.162%p |

## `[-8,8)`

| variant | TFLOP/s mean +/- SD | paired delta vs baseline |
|---|---:|---:|
| `baseline` | 1627.704 +/- 2.642 | +0.000% +/- 0.000%p |
| `no_memset` | 1628.036 +/- 0.747 | +0.021% +/- 0.179%p |
| `sink_keep` | 1627.583 +/- 1.810 | -0.007% +/- 0.197%p |
| `sink_trim` | 1631.685 +/- 5.151 | +0.245% +/- 0.329%p |
| `fixed` | 1624.173 +/- 3.830 | -0.217% +/- 0.369%p |
| `suspend` | 1619.320 +/- 4.914 | -0.515% +/- 0.422%p |
| `fixed_sink` | 1622.183 +/- 3.361 | -0.339% +/- 0.075%p |
| `all` | 1623.293 +/- 3.900 | -0.271% +/- 0.390%p |
