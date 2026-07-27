# 8K/16K/32K N-split library comparison

Date: 2026-07-27

## 결과

| size | BF16 input | ours (static `8x16`) | cuBLAS | selected CUTLASS | ours/cuBLAS |
|---:|---|---:|---:|---:|---:|
| 8K | `[0,1)` | **1772.068 +/- 3.328** | **1807.121 +/- 0.902** | **1667.820 +/- 2.070** | **98.060%** |
| 8K | `[-8,8)` | **1594.164 +/- 1.235** | **1613.517 +/- 1.106** | **1471.757 +/- 26.765** | **98.801%** |
| 16K | `[0,1)` | **1835.421 +/- 0.961** | **1902.320 +/- 1.022** | **1443.860 +/- 0.826** | **96.483%** |
| 16K | `[-8,8)` | **1624.303 +/- 3.847** | **1685.414 +/- 1.087** | **1311.863 +/- 7.713** | **96.374%** |
| 32K | `[0,1)` | **1602.280 +/- 21.458** | **1617.230 +/- 18.678** | **1282.530 +/- 2.216** | **99.076%** |
| 32K | `[-8,8)` | **1391.279 +/- 9.220** | **1420.659 +/- 5.608** | **1136.013 +/- 3.323** | **97.932%** |

단위는 TFLOP/s이고 각 값은 세 독립 프로세스의
`mean +/- sample SD`다. 32K `[0,1)`의 ours와 cuBLAS는 process 간
변동이 각각 21.458/18.678 TFLOP/s이므로 0.924% 평균 차이를 확정적인
kernel 차이로 해석하지 않는다.

## 조건

| 항목 | 설정 |
|---|---|
| problem | square row-major `C=A@B`, 8K/16K/32K |
| datatype | BF16 A/B, FP32 accumulation/output |
| input | 같은 deterministic seeds의 uniform `[0,1)`, `[-8,8)` |
| timing | cell당 별도 프로세스, warmup 1회, timed 5회 |
| samples | 각 cell 독립 프로세스 3개 |
| ordering | ours/cuBLAS/CUTLASS cyclic rotation; 각 방법이 각 위치에 1회 |
| GPU | NVIDIA B200, driver 580.126.09 |
| toolkit/library | CUDA 12.9.86, cuBLAS 12.9.1.4 |

Ours는 세 크기 모두 148 persistent CTA, static `8x16` M-fast,
diagnostic sink 제거, no suspend다. Size별 compile-time performance
shape는 다음과 같다.

| size | template `<ktiles, mtile_count, ntile_count>` |
|---:|---|
| 8K | `<128,32,32>` |
| 16K | `<256,64,64>` |
| 32K | `<512,128,128>` |

세 binary 모두 512 pattern/ones full-C CPU-reference validation을 zero
error로 통과했다.

8K CUTLASS는 이전 sweep에서 선택한 Stream-K kernel을 사용했다.
16K/32K는 selected CLC `256x256x64`, static cluster `4x1x1`,
five-stage kernel을 사용했다. 두 binary 모두 CUTLASS
`e8ecfad75b44d1ad56264f5001d877e9e47fe080` 4.6.0-dev snapshot과
local benchmark patch 기반이며 최신 CUTLASS 전체의 성능 상한이라는
주장은 아니다.

## 개별 측정값

전체 18개 cell의 세 process 값은
[aggregate.csv](../results/gemm_size_library_compare_20260727_f417348/aggregate.csv)에
보존했다. 실행 순서는
[sequence.tsv](../results/gemm_size_library_compare_20260727_f417348/sequence.tsv)에
있다.

## 재현과 artifact

실험 정의 commit은 `f417348`이다.

```bash
./run_b200_gemm_size_library_compare.sh \
  /workspace/benchmark /workspace/gemm_size_library_compare
```

- [raw result directory](../results/gemm_size_library_compare_20260727_f417348/)
- [generated summary](../results/gemm_size_library_compare_20260727_f417348/summary.md)
- archive SHA-256:
  `7dcdeabd55ab126308f7e903a6e3c32f1ed28cca34ff1c3f505ce39fa3cc2198`
