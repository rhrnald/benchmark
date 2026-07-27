# 16K N-split library comparison

Date: 2026-07-27

## 결과

| BF16 input | ours (`fixed_sink`) | cuBLAS | selected CUTLASS CLC | ours/cuBLAS |
|---|---:|---:|---:|---:|
| uniform `[0,1)` | **1834.342 +/- 2.432** | **1897.885 +/- 3.031** | **1443.285 +/- 1.724** | **96.652%** |
| uniform `[-8,8)` | **1623.016 +/- 1.993** | **1683.923 +/- 2.435** | **1305.937 +/- 5.014** | **96.383%** |

단위는 TFLOP/s이며 `mean +/- sample SD`다. 현재 canonical은 cuBLAS의
약 96.4--96.7%이고, 이 비교에서 사용한 selected CUTLASS kernel보다는
두 분포 모두 빠르다.

## 동일 조건

| 항목 | 설정 |
|---|---|
| problem | row-major `C[16384,16384] = A @ B` |
| datatype | BF16 A/B, FP32 accumulation/output |
| input | 같은 deterministic seeds의 `[0,1)`, `[-8,8)` |
| timing | cell당 별도 프로세스, warmup 1회, timed 5회 |
| samples | 각 cell 독립 프로세스 6개 |
| ordering | 세 방법의 6개 순열을 전부 사용; 각 방법이 각 위치에 2회 |
| GPU | NVIDIA B200, driver 580.126.09 |

Ours는 static `8x16` M-fast, 148 persistent CTA의 `fixed_sink`
canonical이다. 측정 source SHA-256은
`eb90e11322c3af9adba08ed6262fe585bd0b37c9ba2fb1750c802e393c6b6dc2`다.
512 pattern/ones full-C CPU-reference validation을 모두 zero error로
통과했다.

cuBLAS는 CUDA 12.9.1.4의 `libcublas.so.12`와
`libcublasLt.so.12`를 사용했다. 실행 로그의 `cublasGetVersion()`은
`120901`이다.

CUTLASS는 이전 sweep에서 16K 대상으로 선택했던 CLC kernel이다.
Base revision은 `e8ecfad75b44d1ad56264f5001d877e9e47fe080`이며,
`256x256x64`, static cluster `4x1x1`, five-stage, CLC scheduler를
사용한다. Binary SHA-256은
`5c993a7d1bf77ee7c5e1bbe2568cbc0bf965f23d592a53d60ab72fe16eeaf754`다.
이는 CUTLASS 4.6.0-dev snapshot과 local benchmark patch로 만든 선택
kernel이며, 최신 CUTLASS 전체에서 가능한 최댓값이라는 주장은 아니다.

## 개별 측정값

| input | method | six process TFLOP/s samples |
|---|---|---|
| `[0,1)` | ours | 1835.508, 1837.485, 1836.410, 1832.551, 1832.456, 1831.640 |
| `[0,1)` | cuBLAS | 1901.551, 1897.330, 1892.757, 1898.564, 1897.052, 1900.058 |
| `[0,1)` | CUTLASS | 1442.770, 1440.240, 1444.910, 1444.320, 1442.960, 1444.510 |
| `[-8,8)` | ours | 1623.586, 1623.058, 1626.437, 1620.456, 1622.589, 1621.972 |
| `[-8,8)` | cuBLAS | 1680.956, 1682.538, 1683.868, 1684.594, 1683.395, 1688.188 |
| `[-8,8)` | CUTLASS | 1306.890, 1307.130, 1296.780, 1306.890, 1312.150, 1305.780 |

## 재현과 artifact

실험 정의 commit은 `34846b0`이다.

```bash
CUTLASS_BIN=/workspace/gemm_compare_bins/cutlass_bf16_best_clc_bench \
  ./run_b200_gemm_16k_library_compare.sh \
  /workspace/benchmark /workspace/gemm_16k_library_compare
```

- [raw result directory](../results/gemm_16k_library_compare_20260727_34846b0/)
- [generated summary](../results/gemm_16k_library_compare_20260727_34846b0/summary.md)
- archive SHA-256:
  `405ed2aed612044c3c2848047c45ac303d96f6f3999934237c8e3c67f3fb48e4`
