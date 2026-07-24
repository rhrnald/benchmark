# 16K GEMM optimization plan

## 목표와 기준선

목표는 단순히 L2 hit rate만 올리는 것이 아니라, 실제 위치의 dense
`16384 x 16384 x 16384` GEMM에서 현재 드러난 모든 serial bottleneck을
분리해 찾는 것이다. 첫 기준선은 `gemm256_bf16_16k.cu`이며, 보존된
`p0` binary의 1806.657 TFLOP/s를 같은 GPU에서 재현하는 것이 역사적 B0
gate였다. 이 gate와 E2a를 통과한 뒤 채택한 E7a dual-wide u1이 현재 후속
실험의 source-backed baseline이다.

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

새 GPU, driver, CUDA/cuBLAS library 또는 Vast host로 옮길 때는 pinned
cuBLAS 실행 파일을 16K의 두 input distribution에서 같은 1/5 규약으로
3프로세스 측정한다. absolute TFLOP/s와 함께 custom/cuBLAS 비율을 환경
calibration으로 기록한다. 이 비율은 서로 다른 환경의 결과를 참고용으로
비교하는 데만 쓰며, 코드 채택 여부는 계속 같은 세션의 baseline/candidate
paired delta로 결정한다.

instance `45481495`의 calibration은 `[0,1)`에서 `p0 1737.130`, E7a
`1770.382`, cuBLAS `1836.906 TFLOP/s`였고 E7a/cuBLAS는 96.378%였다.
`[-8,8)`은 각각 1512.169, 1529.712, 1608.169 TFLOP/s와 95.121%였다.
과거 `p0 1806.657` 환경에는 같은 세션의 cuBLAS 측정이 없으므로,
cross-instance 정규화는 직접 증거로 사용하지 않는다.

같은 instance와 software stack의 size extension에서 현재 E7a/cuBLAS는
`[0,1)` 기준 8K 98.104%, 32K 98.939%였고, `[-8,8)` 기준 confirmation
8K 98.434%, 32K 100.505%였다. 32K signed의 차이는 cuBLAS process
변동보다 작으므로 동률로 취급한다. 32K에서는 `16x16` direct port보다
`8x18` macro가 `[0,1)` +0.791%, `[-8,8)` +4.735%여서 size extension의
기준으로 사용한다. 이 수치는 환경 calibration용이며 16K 후보 채택
gate에는 사용하지 않는다.

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

### E5. producer warp 역할 균형 (보류)

이전 E2a phase trace에서 producer completion skew는 `[0,1)` 89 cycle,
`[-8,8)` 13 cycle뿐이었고 두 producer 모두 consumer보다 약 2.7K cycle
먼저 끝났다. E7a 채택 뒤에는 warp 1이 early B0 16 KiB를 먼저 issue하고,
warp 0이 A 32 KiB 뒤 late B1 16 KiB를 issue한다. 새 phase trace에서
consumer가 B1을 실제로 기다리는 구간이 확인될 때만 다음 producer mapping
ablation을 연다. TMA byte 수와 descriptor shape는 고정한다.

### E6. 다음 task ID와 first-K64 선취 (완료, 미채택)

N-split scalar-x64에서 warp 0의 producer tail이 다음 valid task를 먼저
claim하고, 다음 tile의 정상 kt0 stage를 회전 예약했다. A `256x64`와
B0/B1 `64x128` TMA를 현재 epilogue의 첫 C-store commit/wait 사이에
배치한 C를, 동일 issue를 epilogue 뒤에 둔 B와 비교했다. 두 버전은 같은
두 noinline call과 같은 opcode multiset/resource를 가지며 active call
위치만 다르다.

Matched C/B는 `[0,1)` `+0.0606% [-0.0120,+0.1332]`,
`[-8,8)` `-0.1353% [-0.4209,+0.1504]`였다. 두 입력 gate를 모두
실패했으므로 epilogue-overlap first-K64 prefetch는 미채택한다. B/A의
평균은 약 `+0.25%`였지만 signed CI가 0을 걸치고 여러 구조 변경이 합쳐진
diagnostic이므로 별도 채택 근거로 쓰지 않는다. 세부 내용은
[`../NSPLIT_CROSS_TILE_PREFETCH.md`](../NSPLIT_CROSS_TILE_PREFETCH.md)에
있다.

### E7. consumer MMA topology와 K-loop codegen (완료)

두 warp가 N128씩 `m128n128k16`을 issue하던 기준선을, 두 warp가 각각
M128씩 `m128n256k16`을 issue하는 구조로 바꿨다. B stage는 총 byte 수를
유지한 채 N-split에서 K-split 16 KiB 두 transaction으로 바꾸고, early
B0 뒤 첫 두 MMA와 late B1 뒤 나머지 두 MMA를 issue한다. E7a는 두 분포에서
각각 +1.337%, +1.290%로 채택했다. 동일 코드에서 consumer K loop의
compiler auto-unroll만 허용한 E7b는 -0.027%, -0.209%여서 u1을 유지한다.

동적 MMA issue가 K64당 16회에서 8회로 바뀌었으므로 E2a에서 측정한 P0b
store-off 상한과 P0c phase 비율을 그대로 사용하지 않는다. 다음 비-L2
작업은 E7a default의 same-binary store-off와 phase trace를 다시 측정해
mainloop wait/issue, final drain, epilogue, scheduler 비율을 갱신하는 것이다.

### P1. dual-wide 병목 재측정 (완료)

same-binary C-store-off 상한은 `[0,1)` +3.191%, `[-8,8)` +3.298%였다.
phase trace에서 epilogue는 tile cycle의 3.607%/3.638%, scheduler는
0.300%/0.290%였다. 두 producer는 consumer보다 약 2.9K cycle 먼저 끝났고
producer 간 차이는 34/35 cycle뿐이었다. 반면 consumer의 stage별 A, B0,
B1 readiness 경로 aggregate가 각각 약 37--41K, 26K, 22K cycle이었다.

따라서 다음 순서를 사용한다.

1. 두 consumer commit이 arrival count 2로 합류하는 stage별 공용
   `mma_done` barrier를 독립 측정한다.
2. 두 consumer가 공통으로 필요로 하는 A+B0를 arrival count 2의 공용
   readiness barrier로 합치되 B1 dependency는 추가하지 않는다.
3. 채택된 barrier 변경만 합친다.
4. producer K loop에만 u1을 적용해 code size 감소와 동적 산술 비용을
   독립 비교한다. E8c에서 main-kernel instruction이 1952에서 1144로
   줄었지만 `[0,1)` -0.491%, `[-8,8)` -0.113%여서 채택하지 않았다.
   producer code size는 현재 critical path가 아니다.
5. 이후 epilogue overlap을 cross-tile first-K64 prefetch로 다시
   검토했다. C/B가 `+0.0606%/-0.1353%`여서 미채택했으며, 세부 결과는
   위 E6에 기록했다.

E8d에서는 CUTLASS와 같이 모든 `mbarrier.try_wait`에 `0x989680` suspend
hint를 넣었다. 두 분포의 여섯 pair가 모두 빨랐지만 +0.285%/+0.241%로
0.5% gate 아래였다. broad 후보는 채택하지 않고, 짧은 consumer readiness
wait의 wake-up 비용을 분리하기 위해 producer `mma_done` reuse wait에만
hint를 적용한 E8e를 비교했다. E8e는 12/12 pair가 양수였지만 6쌍 평균이
+0.357%/+0.488%로 역시 두 분포 모두 gate 아래였다. suspend 계열은
positive diagnostic으로 보존하되 default에는 합치지 않는다. 다음은 가장
긴 A readiness를 직접 줄이는 split-A를 clean default에서 독립 측정했다.
E8f는 K64당 총 byte를 유지했지만 TMA transaction이 3개에서 4개로 늘면서
-0.143%/-0.047%였으므로 기각한다. alternate issue order는 측정하지 않는다.
E8g는 TMA 수를 유지한 채 producer를 A 32 KiB/B 32 KiB로 재배치했지만
-0.108%/-0.324%였고 6/6 pair가 느렸다. B0/B1을 한 warp에 직렬화하지 않고
현재의 두 독립 B producer stream을 유지한다.

E9에서는 과거 N-split의 one-time phase 상수를 재사용하지 않고 현재
dual-wide E7a의 measured slack에 맞춰 phase를 다시 설계했다. 148 CTA를
4/8 cohort로 한 번만 stagger한 후보는 `[0,1)`/`[-8,8)`에서
`+0.001%/+0.079%`, `-0.113%/-0.058%`였다. 매 K64에서 A issue 뒤 late
B1만 32/64 cycle suspend한 후보는 exact baseline 대비
`-0.094%/-0.199%`, `-0.118%/-0.253%`였다. W3의 기존 B1 wait를 첫 MMA
앞으로 옮긴 후보도 `-0.146%/-0.002%`였다. B1 codegen control의 한
`[0,1)` outlier를 보존하고 3 pass를 추가한 6-process 결과에서도
B1 32/64 후보는 양쪽 분포 모두 exact baseline보다 느렸다. 모든 후보가
두 분포 `+0.5%` gate를 실패했으므로 phase `0/0`을 유지하고 이 방향을
닫는다. 세부 결과는 [`../E7A_PHASE_REDESIGN.md`](../E7A_PHASE_REDESIGN.md)에
있다.

### 다음 비-L2 순서

1. 동일 TMA byte, 동적 MMA 8회, TMEM destination을 고정한 same-address
   microbenchmark에서 2-consumer와 strict 1-consumer issue를 먼저 비교한다.
   1-consumer는 warp 2가 각 K16에서 M0/M1을 연속 issue하고 wait를
   stage당 6회에서 3회, commit을 2회에서 1회로 줄인다. 총 work가 달랐던
   과거 1-warp/2-warp 숫자는 이 판정에 사용하지 않는다.
2. microbenchmark가 0.5% 안이면 dense kernel로 옮기고, 그보다 크게
   느리면 dense 측정 없이 종료한다.
3. epilogue는 same-binary `stage-only/no-TMA-store` control로 TMEM staging과
   store completion 비용을 먼저 분리한다. 그 뒤에만 세 buffer로 chunk
   0--2를 한 group, chunk 3을 다음 group으로 처리하는 `3+1`을 측정한다.
4. scheduler는 phase trace 상한이 0.3%이므로 위 실험 뒤에도 후순위다.

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
- 단일 32 KiB wide-B TMA와 `CSTORE_CHUNK_N=256` (현재의 K-split
  16 KiB wide-B TMA 두 번은 유지)
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
| E2c | combine A+B0 readiness barrier | exact | 174/0 | 1737.873 | 1491.975 | -0.615% / -1.723% vs macro-free E2a | reject; extra pipe-1 dependency |
| E4a | two-buffer, per-chunk C-store commit/read wait | exact | 172/0 | 1740.561 | 1512.396 | -0.457% / -0.162% vs macro-free E2a | reject; all pairs negative |
| E4b | three-buffer, per-chunk C-store commit/read wait | exact | 172/0 | 1739.175 | 1511.237 | -0.503% / -0.235% vs macro-free E2a | reject; all pairs negative |
| E4c | aligned dynamic-shared declaration / native shared stores | exact | 166/0 | 1735.996 | 1508.330 | -0.713% / -0.380% vs macro-free E2a | reject; all pairs negative |
| E4d | x128 TMEM epilogue loads | exact | 174/0 | 1748.644 | 1514.628 | +0.020% / -0.307% vs macro-free E2a | reject; distribution-dependent |
| E7a | two-warp M-split `m128n256k16`, staggered split-K B TMA, K-loop u1 | exact | 172/0 | 1773.523 | 1531.740 | +1.337% / +1.290% vs macro-free E2a | adopt as clean working default |
| E7b | compiler auto-unroll of dual-wide consumer K loop | exact | 172/0 | 1772.807 | 1527.173 | -0.027% / -0.209% vs dual-wide u1 | reject; all pairs favor u1 |
| P1a | E7a same-binary C-store-off strict ceiling | on exact; off no C | 172/0 | 1830.992 | 1580.081 | +3.191% / +3.298% vs store-on | diagnostic upper bound; epilogue alone cannot close gap |
| P1b | E7a block-0 tile-8 phase trace, five processes | exact before trace | 186/0 diagnostic | N/A | N/A | epilogue 3.607% / 3.638%; scheduler 0.300% / 0.290% | prioritize readiness/completion barrier ablations |
| E8a | shared stage `mma_done`, arrival count 2 | exact, pattern/ones x3 | 170/0 | 1769.467 | 1530.390 | -0.079% / +0.043% vs dual-wide u1 | neutral/reject; producer waits are off critical path |
| E8b | shared A+B0 readiness, arrival count 2; B1 independent | exact, pattern/ones x3 | 172/0 | 1773.802 | 1529.112 | +0.228% / +0.062% vs dual-wide u1 | neutral; below 0.5% gate, do not adopt |
| E8c | producer K loops only `unroll 1` | exact, pattern/ones x1 | 172/0 | 1761.532 | 1526.575 | -0.491% / -0.113% vs dual-wide u1 | reject; producer code size is off critical path |
| E8d | CUTLASS-style suspend hint on all `mbarrier` waits | exact, pattern/ones x1 | 172/0 | 1774.453 | 1534.989 | +0.285% / +0.241% vs dual-wide u1 | neutral; 6/6 positive but below 0.5% gate |
| E8e | CUTLASS suspend hint on producer `mma_done` waits only | exact, pattern/ones x1 | 172/0 | 1777.055 | 1535.769 | +0.357% / +0.488% vs dual-wide u1 | neutral; 12/12 positive but both below 0.5% gate |
| E8f | split A 32 KiB into two M128 x K64 16 KiB TMAs | exact, pattern/ones x1 | 174/0 | 1766.954 | 1525.982 | -0.143% / -0.047% vs dual-wide u1 | reject; fourth TMA transaction does not repay finer readiness |
| E8g | producer ownership A-only / B0-then-B1 | exact, pattern/ones x1 | 176/0 | 1767.832 | 1525.960 | -0.108% / -0.324% vs dual-wide u1 | reject; keep two independent B producer streams |
| E9a | one-time 4-cohort CTA startup staggering | exact, pattern/ones x1 | 172/0 | 1819.517 | 1614.732 | +0.001% / +0.079% | neutral; below gate |
| E9b | one-time 8-cohort CTA startup staggering | exact, pattern/ones x1 | 172/0 | 1817.449 | 1612.529 | -0.113% / -0.058% | reject |
| E9c | per-K64 late-B1 32-cycle spacing | exact, pattern/ones x1 | 180/0 | 1817.785 | 1610.263 | -0.094% / -0.199% vs exact E7a | reject; six-process extension also negative |
| E9d | per-K64 late-B1 64-cycle spacing | exact, pattern/ones x1 | 180/0 | 1817.348 | 1609.389 | -0.118% / -0.253% vs exact E7a | reject; six-process extension also negative |
| E9e | W3 existing B1 wait before first MMA half | exact, pattern/ones x1 | 174/0 | 1816.851 | 1613.433 | -0.146% / -0.002% | reject |
