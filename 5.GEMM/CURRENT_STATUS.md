# GEMM current status

Last updated: 2026-07-25

## 요약

- 현재 최적화 working source는
  [`baseline/gemm256_bf16_16k.cu`](baseline/gemm256_bf16_16k.cu)의
  **E2a N-split** 커널이다. A `256x64`를 두 consumer가 공유하고,
  B0/B1은 각각 `64x128`, 두 warp는 각각 `256x128` C를 누적한다.
- 현재까지 source-backed 성능 최선은 직전 **E7a dual-wide**이며 exact
  source는 결과 artifact와 Git 이력에 보존돼 있다. N-split을 다시
  최적화하되 같은-session E7a exact를 성능 reference로 함께 측정한다.
- fresh 같은-session 결과는 N-split **1797.175/1593.776**, E7a
  **1817.023/1612.664 TFLOP/s** (`[0,1)`/`[-8,8)`)다. N-split은
  각각 1.0924%/1.1712% 낮았다. 첫 producer-suspend 후보는 N-split 대비
  -0.0224%/+0.4660%라 미채택이다.
- 두 번째 fresh 세션에서 exact N-split은 **1795.604/1594.047**,
  E7a는 **1817.047/1608.067 TFLOP/s**였다. Pipe 특수화 control은
  exact N-split보다 **-2.2181%/-1.4136%**, 동일 코드 형태의
  `tcgen05.mma.ws` B-collector 후보는 control보다
  **-4.1799%/-3.7104%**였다. 두 방향 모두 미채택이며 canonical은
  runtime-pipe ordinary-MMA N-split을 유지한다.
- `C_p^T=B_p^T A^T` transpose-compute는 원하는 A `256x64` 공유와
  B0/B1 `64x128`, warp별 logical `256x128` 소유를 유지하면서 K64당
  MMA를 16→8회로 줄였다. No-store mainloop는 **+0.8963%/+0.4722%**,
  scalar-transpose E2E는 **+0.4617%/-0.0792%**였다. Mapping은 유효하지만
  scalar epilogue가 0.019/0.029 ms를 추가해 아직 미채택이다.
  이어서 수행한 vectorized epilogue 결과는 아래와 같다.
- Vectorized epilogue local gate는 naive vec2의 2-way shared-bank
  conflict를 확인했다. Conflict-free x32 vec2는 111/116 registers,
  fused x64 vec2/vec4는 128 registers이며 모두 spill/local 0이다.
  8-position/first-order Williams-balanced B200 실험에서 scalar
  transpose는 exact 대비 **+0.4473%/+0.5994%**였지만 `[0,1)`가
  `+0.5%` adoption gate 아래였다. 모든 vec2/vec4 후보는 scalar보다
  느렸고, 최선 CF2 x64도 **-0.0716%/-0.0020%**였다. Vector 후보는
  미채택이며 세부 결과는
  [`NSPLIT_EPILOGUE.md`](NSPLIT_EPILOGUE.md)에 있다.
- Scalar TMEM-load x64→x32 ablation은 registers를 174→91로 줄였지만
  x64 대비 **-0.0501%/-0.0652%**였고, paired 95% CI도 각각
  `[-0.1521%,+0.0520%]` / `[-0.2287%,+0.0984%]`로 0을 걸쳤다.
  Scalar x64 자체도 exact 대비 **+0.4759%/+0.4762%**로 두 입력
  `+0.5%` adoption gate에 못 미쳤다. 따라서 x32는 미채택하고,
  x16은 측정하지 않으며 direct exact가 canonical이다. 세부 결과는
  [`NSPLIT_SCALAR_TMEM.md`](NSPLIT_SCALAR_TMEM.md)에 있다.
- Cross-tile first-K64 prefetch는 남는 64 KiB stage를 회전 예약하고
  다음 A `256x64`와 B0/B1 `64x128`을 현재 epilogue의 첫 TMA-store
  group과 겹쳤다. Matched C/B는 `[0,1)`에서
  **+0.0606% `[-0.0120,+0.1332]`**, `[-8,8)`에서
  **-0.1353% `[-0.4209,+0.1504]`**여서 overlap을 미채택했다.
  [`NSPLIT_CROSS_TILE_PREFETCH.md`](NSPLIT_CROSS_TILE_PREFETCH.md)에
  설계, 전체 A/B/C 결과와 artifact가 있다.
- Direct N-split phase shift는 prefetch 없이 CTA startup staggering,
  warp-1 B1 issue gap, pipe-1 consumer gap을 측정했다. CTA staggering은
  음수, pipe-1 gap은 control부터 -2.8%~-3.5%로 크게 느렸고,
  유일하게 양수인 `b1_gap64`도 matched control 대비
  **+0.0926%/+0.0681%** (`[0,1)`/`[-8,8)`)에 그쳐 미채택했다.
  [`NSPLIT_PHASE_SHIFT.md`](NSPLIT_PHASE_SHIFT.md)에 결과가 있다.
- Clean N-split A-locality scheduler sweep은 같은-A N neighbors를 먼저
  실행하는 `nfast_16x16`, 더 강한 `8x32`, `4x64` macro를 측정했다.
  `nfast_16x16`은 **+0.6803%/+0.1993%**로 signed input gate를 못 넘었고,
  `8x32`와 `4x64`는 크게 느려졌다. 단순 macro order만으로는 repeated-A
  diagnostic의 +6.5%를 회수하지 못한다.
  [`NSPLIT_A_LOCALITY.md`](NSPLIT_A_LOCALITY.md)에 결과가 있다.
- 최근 direct N-split canonical을 8K/32K로 직접 포팅하고 signed8에서
  size별 macro를 다시 선택했다. W1/I5 프로세스 3회 기준 ours는
  **8K 1568.114 (`8x18`) / 16K 1600.626 (`16x16`) / 32K 1406.844
  (`8x18`) TFLOP/s**였다. 같은 세션 cuBLAS는
  **1610.837 / 1683.090 / 1422.276**, CUTLASS targeted kernel은
  **1485.807 / 1309.717 / 1138.963 TFLOP/s**였다.
  [`NSPLIT_SIGNED8_SIZE_COMPARE.md`](NSPLIT_SIGNED8_SIZE_COMPARE.md)에
  size-port 정의, macro sweep, full-C 검증과 결과가 있다.
- 같은 recent N-split에서 dynamic atomic queue와 static 148-CTA
  grid-stride ownership을 `4x16`, `8x16`, `4x32`, `8x18`, `12x12`,
  `16x16` macro로 signed8 재측정했다. 최고는 **8K static `8x16`
  1592.794**, **16K static `8x16` 1628.954**, **32K dynamic `8x16`
  1400.041 TFLOP/s**였다. 144-task macro가 148 workers에 가깝다는
  이유만으로 유리하지 않았고, static `8x18/12x12`는 padding과 fixed
  ownership의 phase drift 때문에 크게 느려졌다. 우선 16K canonical은
  측정된 signed8 최선인 static `8x16`으로 고정했다. `[0,1)` 확인은
  이 고정 정책의 분포 민감도 확인으로 남아 있다.
  [`NSPLIT_SCHEDULER_SWEEP.md`](NSPLIT_SCHEDULER_SWEEP.md)에 전체 표와
  static load-balance 해석이 있다.
- 이 커널은 repeated-address microbenchmark가 아니라 실제 A/B 좌표를
  읽고 FP32 C 전체를 저장하는 dense end-to-end GEMM이다.
- 직전 E7a의 historical paired 측정은 `[0,1)` **1773.523 TFLOP/s**,
  `[-8,8)` **1531.740 TFLOP/s**다. 별도 fresh B200 세션의 `[0,1)`
  결과는 **1800.000 +/- 1.632 TFLOP/s**였다. 2026-07-24 phase
  ablation 세션의 exact-source baseline은 **1819.500 +/- 0.641** /
  **1613.465 +/- 1.236 TFLOP/s**였다. 세션 간 절대값 대신 같은 세션의
  paired ratio를 최적화 판정에 사용한다.
- 과거 보존 binary `p0`의 절대 최고는 **1806.657 TFLOP/s**였지만,
  당시 exact source object는 남아 있지 않다. 직전 E7a는 Git/result
  artifact에 보존된 historical performance reference이고, 현재
  working canonical은 N-split이다.
- 최근 표준은 한 프로세스당 한 case, warmup 1회, timed 5회다. 다른
  B200이나 activation의 절대 TFLOP/s를 직접 ablation으로 섞지 않고,
  같은 세션의 paired ratio를 우선한다.
- 현재 library 결과는 최신 stable software stack으로 측정한 값이 아니다.
  논문용 최종 표는 CUDA 13.3 Update 1, cuBLAS 13.6.0.2,
  CUTLASS 4.6.1에서 다시 측정해야 한다.

## 현재 N-split optimization kernel

현재 기준 구현은 broad experiment source인
`gemm256_tma_tcgen05_bench.cu`가 아니라, 실험용 macro를 제거해 정리한
16K 전용 source다. 이 파일은 요청한 N-split 방향으로 과거 검증본
`3d2d0a4`와 바이트 단위로 일치하도록 복구했다.

| 항목 | N-split 구성 |
|---|---|
| 문제 | row-major `C[16384,16384] = A @ B`, BF16 A/B, FP32 accumulate/output |
| CTA output | `256 x 256` |
| K stage | `K=64`, SMEM 3-stage, dynamic SMEM 197632 B |
| TMA / K64 | A `256x64` 32 KiB 1회, B0/B1 `64x128` 16 KiB씩 |
| MMA | `m128n128k16`; K64당 CTA 동적 issue 16회 |
| warp 0 | 공유 A와 B0 TMA |
| warp 1 | B1 TMA |
| warp 2/3 | 각각 왼쪽/오른쪽 `256x128` output을 N-split 계산 |
| epilogue | 네 `128x128` FP32 chunk, SW128 SMEM staging 후 TMA store |
| scheduler | 148 persistent CTA, fixed static grid-stride ownership |
| tile order | 16K `8x16` macro, macro N-fast, macro 내부 M-fast |
| phase/cache | TMA 0, MMA 0; promotion 없음; multicast 없음 |
| codegen | REG 174, stack/local/spill 0 |

각 CTA는 scheduler가 정한 실제 `(tile_m, tile_n)`에 대해 모든 K stage의
실제 A/B 주소를 읽고 실제 C 위치에 저장한다. 512 pattern/ones full-C
CPU-reference validation을 bit-exact로 통과했다.

## Benchmark timeline

아래 단계는 서로 다른 workload다. 모두를 end-to-end GEMM 성능으로
해석하면 안 된다.

| 단계 | 조건 | 핵심 결과 | 의미 |
|---|---|---:|---|
| MMA-only | constant BF16 1.0, TMA/C-store 없음 | 2229.664 | 낮은 switching activity의 compute 상한 |
| MMA-only | BF16 uniform `[0,1)`, TMA/C-store 없음 | 1928.151 | input activity에 따른 power/clock 상한 |
| dependent TMA+MMA | repeated panel, depth 3, wait/commit, C-store 없음, 2 issuers | 1800.166 | TMA readiness를 포함한 ceiling |
| validated repeated GEMM | `128x256x128`, same global tile, FP32 C store | 1800.282 +/- 5.092 | 과거 1797 계열 재현 |
| valid same-address GEMM | `256x256x64`, 148 persistent CTA | 8K 1813.887 / 16K 1949.590 / 32K 1782.437 | L2-resident repeated-address ceiling; dense GEMM 아님 |
| first dense GEMM | 실제 A/B 좌표, 초기 scheduler | 8K 1205.758 / 16K 1356.429 / 32K 1186.012 | 초기 end-to-end baseline |
| corrected dense persistent | async stage-reuse barrier, full FP32 C | 8K 1697.631 / 16K 1777.128 / 32K 1615.074 | serialized completion wait 제거 후 회복 |
| historical `p0` | dense 16K, W1/I5, 3 processes | 1806.657 | exact binary 보존, exact source 유실 |
| historical E7a reference | dual-wide M-split source | 1773.523 / 1531.740 | `[0,1)` / `[-8,8)` source-backed reference |
| fresh historical E7a | 별도 B200, `[0,1)`, 4 processes | 1800.000 +/- 1.632 | 환경 변화 범위와 E7a source 성능 재확인 |
| fresh N-split redesign | A 공유 + B N-split, 실제 A/B/C, W1/I5 x3 | 1797.175 / 1593.776 | 같은 세션 E7a보다 -1.0924% / -1.1712% |
| N-split static/WS ablation | pipe-static ordinary / B collector, 실제 A/B/C, W1/I5 x4 | 1755.775 / 1571.485; 1682.383 / 1513.175 | static은 exact보다 -2.2181% / -1.4136%; WS는 static보다 -4.1799% / -3.7104% |
| N-split transpose compute | B^T x A^T로 MMA 16→8, scalar transpose C store, W1/I5 x4 | E2E 1808.175 / 1598.322; no-store 1882.926 / 1670.525 | exact 대비 E2E +0.4617% / -0.0792%; no-store +0.8963% / +0.4722% |
| N-split vector epilogue | scalar + vec2/vec4 5종, Williams-balanced W1/I5 x8 | scalar 1766.507 / 1574.004; best vector CF2 x64 1765.241 / 1573.960 | scalar는 exact 대비 +0.4473% / +0.5994%; 모든 vector는 scalar 미달 |
| N-split scalar TMEM width | exact/x64/x32 전순열-balanced W1/I5 x6 | exact 1758.255 / 1567.368; x64 1766.621 / 1574.829; x32 1765.736 / 1573.798 | x32는 x64 대비 -0.0501% / -0.0652%; x64도 exact 대비 +0.5% gate 미달 |
| N-split cross-tile kt0 prefetch | scalar-x64 A / non-overlap B / epilogue-overlap C, 전순열-balanced W1/I5 x6 | A 1743.253 / 1558.528; B 1747.851 / 1562.414; C 1748.909 / 1560.297 | C/B +0.0606% / -0.1353%; 두 입력 CI gate 실패로 overlap 미채택 |
| N-split A-locality scheduler | local N-fast macro `16x16`, `8x32`, `4x64`, W1/I5 x4 | baseline 1753.998 / 1521.339; best `nfast_16x16` 1765.930 / 1524.371 | +0.6803% / +0.1993%; signed gate 실패, stronger A-locality macro는 -2.4%~-16.2% |

L2/scheduler 실험에서는 persistent가 normal grid보다 유리했다. 최근
direct N-split sweep에서 static `8x16`이 16K signed8 최선으로 측정되어
현재 canonical은 static 148 CTA, `8x16`, no hint, no multicast,
phase `0/0`이다. 단순 phase shift, wide-B 단일 TMA, A/B L2 promotion,
eviction hint, strip ownership, multicast/cluster-4는 채택하지 않았다.

직전 E7a topology에 맞춰 phase shift도 다시 설계해 측정했다. 148 CTA의
one-time 4/8-cohort startup staggering, 매 K64 stage에서 A issue 뒤 B1을
32/64 cycle 늦추는 방식, W3의 기존 B1 wait를 첫 MMA 앞으로 옮기는
방식을 비교했다. Primary `[0,1)` / `[-8,8)` paired 변화는 각각
`cta4 +0.001%/+0.079%`, `cta8 -0.113%/-0.058%`,
`b1_gap32 -0.094%/-0.199%`, `b1_gap64 -0.118%/-0.253%`,
`b1_cross -0.146%/-0.002%`였다. Outlier 보강 6-process 결과에서도
`b1_gap32/64`는 exact baseline보다 두 분포 모두 느렸다. 어떤 후보도
양쪽 분포 `+0.5%` gate를 넘지 못해 canonical phase는 `0/0`을 유지한다.
세부 설계와 raw artifact는
[`E7A_PHASE_REDESIGN.md`](E7A_PHASE_REDESIGN.md)에 있다.

## Library comparison

### 완전한 matched 3-way reference: pre-E7a custom

아래는 같은 B200 세션에서 세 방법, 세 크기, 두 분포를 모두 맞춘 표다.
각 cell은 독립 프로세스 3개의 평균과 sample SD이고, 각 프로세스는
W1/I5다. 단, custom 열은 직전 E7a가 아니라 그 이전의 recovered
persistent kernel이다.

| input | size | pre-E7a custom | cuBLAS | selected CUTLASS |
|---|---:|---:|---:|---:|
| `[0,1)` | 8K | 1713.253 +/- 0.420 | 1787.475 +/- 2.912 | 1649.037 +/- 2.098 |
| `[0,1)` | 16K | 1783.963 +/- 0.918 | 1873.248 +/- 1.987 | 1433.417 +/- 0.671 |
| `[0,1)` | 32K | 1625.957 +/- 2.371 | 1629.405 +/- 22.572 | 1269.063 +/- 6.500 |
| `[-8,8)` | 8K | 1546.731 +/- 2.572 | 1593.261 +/- 0.909 | 1459.173 +/- 2.580 |
| `[-8,8)` | 16K | 1585.642 +/- 2.353 | 1665.146 +/- 2.631 | 1293.517 +/- 5.059 |
| `[-8,8)` | 32K | 1396.409 +/- 7.926 | 1410.183 +/- 19.077 | 1102.067 +/- 2.386 |

Raw data와 binary hash는
[`gemm_default_compare_1x5_b200_45460466`](../results/gemm_default_compare_1x5_b200_45460466/)
에 있다.

### 직전 E7a와 cuBLAS: same-session

| input | size | E7a | cuBLAS | E7a/cuBLAS |
|---|---:|---:|---:|---:|
| `[0,1)` | 8K | 1704.894 | 1737.851 | 98.104% |
| `[0,1)` | 16K | 1770.382 | 1836.906 | 96.378% |
| `[0,1)` | 32K | 1608.133 | 1625.373 | 98.939% |
| `[-8,8)` | 8K | 1555.989 | 1580.738 | 98.434% |
| `[-8,8)` | 16K | 1529.712 | 1608.169 | 95.121% |
| `[-8,8)` | 32K | 1338.924 | 1332.195 | 100.505%* |

`*` 32K signed 차이는 0.505%이고 cuBLAS process SD가
17.252 TFLOP/s이므로 동률로 해석한다. 8K/16K E7a는 `16x16`,
32K는 size-tuned `8x18` macro를 사용한다.

Artifacts:

- [`gemm_env_cublas_reference_b200_45481495_20260723`](../results/gemm_env_cublas_reference_b200_45481495_20260723/)
- [`gemm_env_cublas_size_extension_b200_45481495_20260723`](../results/gemm_env_cublas_size_extension_b200_45481495_20260723/)

### Fresh 16K E7a와 selected CUTLASS

같은 B200 세션, 동일 deterministic `[0,1)` bytes, W1/I5,
AB/BA/AB/BA 네 프로세스 결과다.

| implementation | TFLOP/s |
|---|---:|
| clean E7a | **1800.000 +/- 1.632** |
| selected CUTLASS | **1423.120 +/- 2.577** |

CUTLASS는 `256x256x64`, static `4x1` cluster, 2-SM MMA, 5-stage,
direct-store CLC 구성이다. E7a가 이 선택 후보보다 26.483% 높았지만,
이는 CUTLASS 전체 configuration space의 최댓값을 증명하지 않는다.

Artifact:
[`gemm_e7a_cutlass_16k_compare_b200_45601332_20260723`](../results/gemm_e7a_cutlass_16k_compare_b200_45601332_20260723/)

직전 E7a에 대해 8K/16K/32K x 두 분포 x cuBLAS/CUTLASS를 한 세션에서
모두 맞춘 표는 아직 없다. 위 세 표를 합쳐 하나의 matched 표로
재작성하지 않는다.

## Software provenance

현재 숫자는 모두 최신 stable stack에서 얻은 것이 아니다.

| component | 실제 측정 환경 | 2026-07-23 latest stable |
|---|---|---|
| CUDA toolkit | CUDA 12.9 계열 | CUDA 13.3 Update 1, nvcc 13.3.73 |
| cuBLAS | same-binary reference에서 12.9.1.4 확인 | 13.6.0.2 |
| CUTLASS | main `e8ecfad75...`, 4.6.0-dev snapshot + local patch | v4.6.1, `e05f953` |
| driver | 580.126.09 | CUDA 13.3 packaged driver 610.43.02 |

세부적으로 matched 3-way의 보존 binaries는 nvcc 12.9.41로 빌드되었고
host에는 nvcc 12.9.86이 설치되어 있었다. Fresh 16K에서는 E7a를
nvcc 12.9.86으로 새로 빌드했지만 CUTLASS binary는 nvcc 12.9.41
build였다. 과거 matched 3-way 세션의 exact cuBLAS runtime patch는
기록하지 않았고, 이후 동일 SHA runner가 12.9.1.4를 load한 것은 별도로
확인했다.

CUTLASS `e8ecfad`는 tagged 4.6.0/4.6.1 release가 아니라
2026-06-26의 4.6.0-dev main snapshot이다. 따라서 기존 숫자를
“최신 CUTLASS” 또는 “CUTLASS 최댓값”이라고 부르지 않는다.

Driver 580.126.09는 CUDA 13.x minor compatibility 범위에 들지만,
CUDA 13.3의 packaged/native driver는 610.43.02다. 논문용 최종 표는
가능하면 610.43.02+ host를 사용하고, 580 host를 사용하면 compatibility
mode임을 명시한다.

Official references:

- [CUDA Toolkit 13.3 Update 1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [CUTLASS v4.6.1 release](https://github.com/NVIDIA/cutlass/releases/tag/v4.6.1)

## 최신 stack 재측정 계획

1. 공식 `nvidia/cuda:13.3.0-devel-ubuntu22.04` image의 CUDA/cuBLAS를
   Update 1 package로 올린다.
2. CUDA 13.3 Update 1에서 E7a, cuBLAS runner, CUTLASS v4.6.1을 모두
   `sm_100a`로 새로 빌드한다.
3. `nvcc --version`, driver, `cublasGetVersion()`, CUTLASS tag/commit,
   source/binary SHA-256과 실제 동적 library 경로를 기록한다.
4. E7a 512 pattern/ones validation과 CUTLASS correctness check를 먼저
   통과시킨다.
5. 8K/16K/32K x `[0,1)`/`[-8,8)` x E7a/cuBLAS/CUTLASS의 18개 cell을
   측정한다.
6. Cell마다 별도 프로세스, W1/I5를 사용하고 세 번의
   position-balanced rotation으로 평균과 sample SD를 구한다.
7. 모든 방법에 동일 deterministic BF16 bytes/seeds, FP32
   accumulation/output, `beta=0`, complete output store와 CUDA-event
   timing 범위를 적용한다.
8. CUTLASS는 기존 selected config를 먼저 port하고, 크기별 targeted
   sweep을 별도 단계로 수행한다. Quick port 결과를 최신 CUTLASS
   best-of라고 부르지 않는다.
9. Artifact를 내려받고 확인한 뒤 Vast instance를 정지한다.

현재 8K CUTLASS Stream-K 수정 source는 로컬 CUTLASS worktree에만
있으므로 v4.6.1 port 전에 patch/source를 먼저 archive해야 한다.

## Reproduce

16K canonical N-split:

```bash
cd 5.GEMM/baseline
make build
make validate
./run_1x5.sh ./gemm256_bf16_16k /tmp/gemm_runs random
./run_1x5.sh ./gemm256_bf16_16k /tmp/gemm_runs random-signed8
```

주요 문서와 artifact:

- 전체 timeline과 generic 실행법: [`README.md`](README.md)
- canonical N-split 설명: [`baseline/README.md`](baseline/README.md)
- current N-split source: [`baseline/gemm256_bf16_16k.cu`](baseline/gemm256_bf16_16k.cu)
- 비-L2 최적화 ledger: [`baseline/OPTIMIZATION_PLAN.md`](baseline/OPTIMIZATION_PLAN.md)
- non-multicast L2 scheduler ledger: [`NON_MULTICAST_L2_EXPERIMENT_PLAN.md`](NON_MULTICAST_L2_EXPERIMENT_PLAN.md)
- E7a 전용 phase redesign: [`E7A_PHASE_REDESIGN.md`](E7A_PHASE_REDESIGN.md)
- N-split cross-tile kt0 prefetch:
  [`NSPLIT_CROSS_TILE_PREFETCH.md`](NSPLIT_CROSS_TILE_PREFETCH.md)
- historical 3-way runner: [`../run_b200_gemm_compare_1x5.sh`](../run_b200_gemm_compare_1x5.sh)
