# 16K GEMM optimization plan

## 목표와 기준선

목표는 단순히 L2 hit rate만 올리는 것이 아니라, 실제 위치의 dense
`16384 x 16384 x 16384` GEMM에서 현재 드러난 모든 serial bottleneck을
분리해 찾는 것이다. 첫 기준선은 `gemm256_bf16_16k.cu`이며, 보존된
`p0` binary의 1806.657 TFLOP/s를 같은 GPU에서 재현하는 것이 B0 gate다.

모든 실험은 다음 조건을 고정한다.

- NVIDIA B200, power limit 1000 W
- BF16 A/B와 FP32 C, 실제 global tile 주소
- primary input BF16 uniform `[0,1)`, secondary input `[-8,8)`
- 한 프로세스당 한 case, warmup 1회, timed 5회 평균
- baseline/candidate 순서를 바꾼 프로세스 3쌍 이상
- 성능 전에 512 CPU-reference validation
- binary SHA-256, Git commit, ptxas register/spill, resource usage 기록
- 측정 전후 temperature, graphics/SM clock, power 기록

paired mean이 0.5% 미만이면 noise로 간주한다. 0.5--1.0% 후보는 표본을
6쌍으로 늘리고, 1.0% 이상이면서 양쪽 input distribution에서 방향이
같은 후보만 기본선에 합친다. 후보 하나마다 source commit과 result commit을
분리해 ablation이 섞이지 않게 한다.

## B0. clean reconstruction 검증

1. local CUDA 12.9 build와 `REG 178 / spill 0`을 확인한다.
2. B200에서 pattern과 ones validation을 실행한다.
3. clean과 보존된 `p0`를 동일 인스턴스에서 AB/BA로 교차 측정한다.
4. SASS의 MMA/TMA/barrier 개수와 kernel resource를 비교한다.

B0가 1% 이내로 재현되지 않으면 아래 최적화로 넘어가지 않는다. 먼저
SASS diff, CUDA toolchain, clock/power, scheduler mapping 순서로 차이를 찾는다.

## P0. phase trace와 상한 control

작은 instruction 제거만으로 병목 위치를 추측하지 않도록, E3을 끝낸 직후
대표 CTA 하나에만 `clock64` trace를 넣는다. warp 0/1 producer, warp 2/3
TMA wait와 MMA issue/final drain, 네 C chunk의 TMEM-to-SMEM staging/TMA store,
task atomic/decode 구간을 각각 분리한다. 별도 store-off control은 최종 MMA
drain은 유지하고 C staging/store만 생략해 epilogue 최적화의 최대 이득 상한을
구한다. trace 소스와 성능 소스는 분리하며 trace 자체의 TFLOP/s는 결과로
사용하지 않는다.

## 비-L2 hot path 실험

### E1. stage ring 산술 전문화

현재 producer와 consumer loop는 매 K stage마다 `% 3`, `/ 3`으로 stage와
phase를 다시 계산한다. `stage = 0,1,2`와 parity를 증분 갱신하는 ring으로
바꾸고, 동일한 TMA/MMA/barrier 순서를 유지한다. 먼저 SASS에서 integer
instruction과 register 수가 실제로 감소했는지 확인한다. compiler가 이미
상수 나눗셈을 충분히 최적화했다면 성능 측정 없이 종료할 수 있다.

### E2. SMEM descriptor 계산 hoist

A/B shared-memory descriptor의 invariant bit와 stage base를 loop 밖에서
만들고, `kk=0..3`의 offset만 직접 사용한다. E1과 섞지 않고 별도로
측정한다. 목표는 MMA issue warp의 scalar dependency chain을 줄이는 것이다.

### E3. epilogue fence/barrier 축소

현재 각 `128 x 128` C chunk마다 staging 뒤 CTA barrier 두 번과 async
proxy fence가 있고, 두 chunk마다 TMA store group을 wait한다. 다음을
각각 독립적으로 검증한다.

1. staging 완료 barrier와 proxy fence에 필요한 참여 범위를 최소화한다.
2. 같은 shared buffer를 다시 쓰기 직전에만 store completion을 wait한다.
3. 불필요한 epilogue 끝 CTA barrier를 하나씩 제거하되 512 full-C
   reference, 반복 validation, 지원되는 Compute Sanitizer 검사를 모두
   통과시킨다. performance 경로의 sink checksum은 correctness 판정에 쓰지
   않는다.

### E4. 3-buffer C-store pipeline

mainloop가 쓰던 196608 B shared payload를 epilogue에서 재사용하므로
`128 x 128 x FP32` C buffer를 3개 둘 수 있다. 먼저 2-buffer에서 chunk별
commit 후 재사용 직전에 `wait_group.read 1`, 다음으로 3-buffer에서 chunk별
commit 후 재사용 직전에 `wait_group.read 2`를 사용하고 마지막에만
`wait_group.read 0`을 수행한다. 이 순서로 2-buffer/3-buffer를 비교한다.
네 chunk 전체를 위한 4-buffer는 현재 dynamic shared-memory 한도를 넘으므로
대상에서 제외한다.

### E5. producer warp 역할 균형

현재 warp 0은 A 32 KiB와 B0 16 KiB를 issue하고, warp 1은 B1 16 KiB만
issue한다. 다음 두 topology를 동일 work로 비교한다.

- baseline: warp 0 = A+B0, warp 1 = B1
- candidate: warp 0 = A, warp 1 = B0+B1

TMA byte 수와 descriptor는 바꾸지 않는다. 효과가 있으면 producer-side
instruction imbalance가 병목이었다고 판단한다.

### E6. 다음 task ID 선취

현재 global atomic task fetch는 이전 tile의 MMA와 C store가 모두 끝난 뒤
시작한다. 현재 tile 좌표를 보존한 채 epilogue 진입 전에 다음 task ID를
선취하고, C store와 sink 기록 뒤 바로 다음 mainloop를 시작하게 한다.
atomic 수와 tile order는 동일하게 유지한다. 별도로 persistent CTA 수
148 고정과 132/140/144/148 sweep을 비교해 tail 효과도 확인한다.

### E7. 1-warp 대 2-warp MMA issue

두 warp가 N128씩 issue하는 기준선과 한 warp가 두 N128을 모두 issue하는
후보를 비교한다. K 반복, TMA bytes, MMA 개수, C store와 scheduler는
완전히 같아야 한다. 이 실험은 warp 수가 아니라 issue-side dependency와
warp specialization의 비용을 분리하기 위한 것이다.

## L2/scheduler 실험

비-L2 실험에서 채택된 변경을 합친 뒤 다음을 진행한다.

1. `16 x 16` macro 주변에서 `8x32`, `16x16`, `32x8`을 비교한다.
2. 각 macro shape에서 local M-fast/N-fast와 macro M-fast/N-fast를 분리한다.
3. dynamic atomic scheduler와 동일 tile order의 static persistent mapping을
   비교한다. static 후보는 CTA별 iteration 수 차이와 padded task를 함께
   기록한다.
4. Nsight Compute로 L2 hit rate뿐 아니라 DRAM bytes, TMA stalls, tensor
   active, barrier stall, issue active를 함께 비교한다.
5. non-multicast 최선이 확정된 뒤에만 2-CTA cluster multicast를 다시
   비교한다. 사용 SM이 144개로 줄어드는 비용을 반드시 포함한다.

## 당장 반복하지 않을 실험

기존 측정에서 효과가 없거나 악화된 항목은 우선순위를 낮춘다.

- TMA/MMA phase shift: 16K에서 `0/0`이 최선
- wide-B와 `CSTORE_CHUNK_N=256`
- K32, stage 4/5
- 단순 L2 promotion hint
- 현재 형태의 static scheduler와 multicast
- 단순 sink 제거, fused C staging, dedicated epilogue warpgroup

이 항목들은 앞선 hot-path 변경으로 병목 구조가 바뀌었다는 profiler 근거가
생긴 경우에만 다시 연다.

## 실험 기록 표

각 실험 결과 문서에는 아래 열을 유지한다.

| commit | change | validation | REG/spill | `[0,1)` TFLOP/s | `[-8,8)` TFLOP/s | paired delta | decision |
|---|---|---|---|---:|---:|---:|---|
| B0 | clean reconstruction | exact | 178/0 | 1738.199 | 1508.979 | -0.052% / -0.225% vs p0 | pass |
| E1 | incremental stage ring | exact | 184/0 | 1729.041 | 1505.076 | -0.535% / -0.403% | reject |
| E3.1 | remove pre-fence C-store barriers | exact | 178/0 | 1739.212 | 1508.947 | +0.033% / +0.057% | neutral |
| E3.2 | TMA store `wait_group.read 0` | exact | 178/0 | 1738.970 | 1508.044 | -0.062% / -0.249% vs E3.1 | neutral/reject |
| E3.3 | remove duplicate final epilogue barrier | exact | 178/0 | 1737.792 | 1508.663 | -0.039% / -0.123% vs E3.2 | neutral/reject |
| P0a | compile-time C-store-off aggressive ceiling | N/A (no C) | 54/0 | 1789.649 | 1549.232 | +2.891% / +2.453% | diagnostic; strict same-binary control required |
| P0b | same-binary C-store-off strict ceiling | on exact; off no C | 178/0 | 1800.517 | 1560.241 | +3.813% / +3.256% vs dual-on | diagnostic upper bound |
| P0c | CTA phase trace at tile iteration 8 | exact before trace | 184/0 diagnostic | N/A | N/A | C epilogue 3.598% / 3.624% of tile | prioritize E4; defer E5/E6 |
| E2a | direct TMEM scalar addressing | exact | 174/0 | 1751.903 | 1518.912 | +0.614% / +0.393% (6 pairs, all positive) | adopt as clean working default |
| E2b | one shared-address conversion per K stage | exact | 174/0 | 1750.583 | 1516.135 | -0.020% / -0.162% vs E2a | reject |
| E4a | two-buffer, per-chunk C-store commit/read wait | exact | 172/0 | 1740.561 | 1512.396 | -0.457% / -0.162% vs macro-free E2a | reject; all pairs negative |
| E4b | three-buffer, per-chunk C-store commit/read wait | exact | 172/0 | 1739.175 | 1511.237 | -0.503% / -0.235% vs macro-free E2a | reject; all pairs negative |
