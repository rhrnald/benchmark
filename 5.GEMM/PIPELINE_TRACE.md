# E7a per-K-stage pipeline trace

이 trace는 `0.attention`의 `clock64()` trace와 같은 질문을 현재 E7a
GEMM에 적용한다.

- TMA 주소/`expect_tx` 준비와 instruction issue를 언제 시작하고 끝냈는가?
- consumer가 A/B ready barrier를 언제 기다리기 시작하고 통과했는가?
- MMA instruction 묶음을 언제 issue하고 commit했는가?
- commit된 MMA가 3-stage ring buffer reuse barrier에서 언제 완료된 것으로
  관측됐는가?
- producer, consumer와 서로 다른 K64 stage가 실제로 얼마나 겹치는가?

기존 `dual_wide_phase_trace`는 output tile 전체와 256회 호출의 합계를
측정하는 coarse trace다. 전체 병목 비중에는 유용하지만 per-stage 순서와
overlap을 복원할 수 없다. 이 문서의 pipeline trace는 clean E7a에서 선택한
짧은 window만 계측하며 두 trace를 서로 다른 목적으로 유지한다.

## Sampling scope

| 항목 | 값 |
|---|---|
| GEMM | `16384 x 16384 x 16384`, BF16 A/B, FP32 C |
| CTA / tile | block 0, ninth valid persistent tile (`tile_iter=8`) |
| core issue window | K64 `kt=56..63` |
| dependency-observation context | `kt=64..66` |
| timestamp writers | 각 physical warp의 lane 0 |
| clock domain | 같은 CTA/SM의 `clock64()` |
| protocol | untraced warmup launch 1회 + traced launch 1회 |

`kt=64..66`을 기록하는 이유는 3-stage ring 때문이다. `kt=x`에서 commit된
MMA는 같은 shared-memory slot을 재사용하는 producer의 `kt=x+3`
`mma_done` wait가 통과할 때 software dependency-pass evidence를 얻는다.
Core `kt=56..63`의 tail observation을 모두 닫으려면 `kt=66`까지 필요하다.
Context 세 stage에서는
W0/W1의 M0/M1 reuse wait 네 개만 계측하고, 후속 TMA/consumer event는
계측하지 않는다.

## Recorded events

각 recorded K64 stage에는 고정 slot 19개가 있다.

| warp | events |
|---|---|
| W0 | `mma_done[M0]` wait, `mma_done[M1]` wait, A TMA prepare+issue, B1 TMA prepare+issue |
| W1 | `mma_done[M0]` wait, `mma_done[M1]` wait, B0 TMA prepare+issue |
| W2 | A wait, B0 wait, first two wide MMA prepare+issue, B1 wait, last two wide MMA prepare+issue, commit |
| W3 | W2와 동일 |

TMA bar는 helper 전체를 감싸므로 주소 계산, `mbarrier.arrive.expect_tx`,
TMA issue가 포함된다. Bar의 끝은 transfer completion이 아니라 issuer
warp의 post-call timestamp다. 같은 `kt`의 W2/W3 ready-wait 종료 중 빠른
시각을 first dependency pass, 느린 시각을 all-consumer pass로 표시한다.

MMA 구간에는 descriptor 준비와 두 `tcgen05.mma` issue가 포함된다. Commit
종료도 tensor-core execution completion이 아니다.
`kt+3`에서 W0/W1이 해당 `mma_done[mblock]` wait를 통과한 시각을
software dependency-pass observation으로 표시한다. 별도의 completion wait는 추가하지
않으므로 원래 dependency graph는 유지된다.

## Generate and build

성능용 clean source에는 trace macro나 branch를 추가하지 않는다. Generator는
audited clean-source SHA-256이 정확히 일치할 때만 별도 trace source를 만든다.

```bash
cd /home/chaewon/benchmark

python3 5.GEMM/generate_gemm_e7a_pipeline_trace.py \
  --base 5.GEMM/baseline/gemm256_bf16_16k.cu \
  --output build/gemm_e7a_pipeline_trace.cu

/usr/local/cuda-12.9/bin/nvcc -std=c++17 -O3 \
  -gencode arch=compute_100a,code=sm_100a -lineinfo -Xptxas=-v \
  build/gemm_e7a_pipeline_trace.cu \
  -o build/gemm_e7a_pipeline_trace -lcuda
```

## Validate and collect

```bash
build/gemm_e7a_pipeline_trace \
  --validate --validate-size 512 --validate-pattern pattern
build/gemm_e7a_pipeline_trace \
  --validate --validate-size 512 --validate-pattern ones

build/gemm_e7a_pipeline_trace \
  --input-init random \
  --pipeline-trace-csv results/pipeline_trace.csv
```

현재 `--validate` 명령은 같은 generated binary의
`pipeline_trace=nullptr` 경로를 검증한다. Timestamp/store가 활성화된 16K
trace launch는 full C reference를 별도로 비교하지 않는다.

Trace 실행시간은 TFLOP/s로 사용하지 않는다.

## Render

```bash
python3 5.GEMM/plot_gemm_e7a_pipeline_trace.py \
  --trace results/pipeline_trace.csv \
  --svg results/pipeline_trace.svg \
  --metrics-csv results/pipeline_trace_metrics.csv
```

SVG는 실제 W0--W3 lane에 prepare/issue와 wait bar를 그리고, issuer의
post-call에서 observer dependency-pass까지의 signed software-observation
window와 first/all marker를 별도로 표시한다. `metrics.csv`에는 per-stage
commit cadence, A/B0/B1 ordered residual wait, M0/M1 reuse wait, signed
post-call gap, issue/prepare-start 기준의 보수적 observation bound가 들어간다.

## Measured result

2026-07-23 B200 실측의 SVG, raw CSV, per-stage metrics, correctness, SASS와
해석은 다음 결과 디렉터리에 보존했다.

[`../results/gemm_e7a_pipeline_stage_trace_b200_45481495_20260723/README.md`](../results/gemm_e7a_pipeline_stage_trace_b200_45481495_20260723/README.md)

핵심적으로 B0/B1 residual wait는 대부분 ready-check 수준으로 숨겨졌지만,
producer는 M0 ring reuse에 강하게 back-pressure되었고 A ready에는
`ring_stage=1`에서 반복적인 bubble이 관측됐다. 이는 한 CTA/window의
instrumented diagnostic이므로, physical slot imbalance라고 확정하려면
window 또는 sampled tile을 옮긴 재현이 필요하다.

## Interpretation limits

- `clock64()`는 warp가 instruction boundary를 실행한 시각이다. DMA나
  tensor-core 내부의 정확한 hardware start/end timestamp가 아니다.
- Ready/reuse barrier 종료는 software dependency-pass observation이며
  정확한 hardware completion edge가 아니다.
- 서로 다른 warp의 post-call stamp와 observer stamp 순서가 뒤집혀 signed
  gap이 음수가 될 수 있다. 이는 음수 hardware latency가 아니다.
- A→B0→MMA B0→B1, M0→M1 순서로 wait하므로 각 wait bar는 독립 latency가
  아니라 앞선 작업과 겹친 뒤 남은 ordered residual wait다.
- Ready 상태의 `mbarrier_wait`도 instruction overhead 때문에 0 cycle이
  아니다.
- Timestamp와 shared trace record 자체가 선택 window와 compiler codegen을
  교란한다. 이 결과는 clean E7a의 무교란 실행 기록이 아니라 instrumented
  E7a diagnostic이다.
- 서로 다른 CTA/SM의 raw `clock64()`는 한 시간축에 합치지 않는다.
