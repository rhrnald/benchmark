# Alternating K64/K128 pipeline clock64 trace

## Scope

- GPU: NVIDIA B200, Vast instance `46380702`.
- Trace tooling commit: `a18e885`.
- Kernel: alternating `K64 -> K128` two-slot experiment.
- Input/shape: BF16 uniform `[0,1)`, `M=N=K=16384`.
- Traced CTA: block 0 on SM 142.
- Traced persistent output tile: `tile_iter=8`, tile `(16,20)`.
- Window: logical stages 56 through 63.
- Consumer order: `wait B -> wait A0 -> wait A1 -> MMA -> commit`.

The trace was collected five times. The total recorded windows were 21,124,
22,777, 23,278, 23,361, and 24,608 cycles. Pass 3 is plotted because its
23,278-cycle window is the median.

## Mean measured intervals in plotted pass

All values are SM clock cycles. They are wall-clock intervals around an issue
sequence or blocking wait, so they include warp descheduling.

| event | K64 | K128 |
|---|---:|---:|
| A0 TMA issue | 65.0 | 65.0 |
| A1 TMA issue | - | 69.0 |
| B0 TMA issue | 91.0 | 91.0 |
| B1 TMA issue | 93.0 | 93.0 |
| W0 reuse wait, pipe 0 | 239.0 | 3110.5 |
| W1 reuse wait, pipe 1 | 1020.0 | 2793.8 |
| W2 wait B0 | 68.5 | 1792.5 |
| W3 wait B1 | 69.0 | 305.5 |
| W2 wait A0 | 84.0 | 111.5 |
| W3 wait A0 | 84.0 | 1353.8 |
| W2 wait A1 | - | 80.0 |
| W3 wait A1 | - | 394.0 |
| W2 MMA issue sequence | 529.0 | 1228.8 |
| W3 MMA issue sequence | 654.8 | 1060.5 |

The TMA issue bars measure instruction issue, not transfer completion. The end
of the matching consumer wait is the first observed completion point.
Likewise, `tcgen05.commit` records completion-barrier issue; the later producer
reuse-wait end observes actual MMA completion.

## Interpretation

The measured timeline shows an uneven cadence:

1. A K64 stage issues one A slab and eight MMA instructions per consumer.
2. The following K128 stage issues two A slabs and sixteen MMA instructions
   per consumer.
3. Since the same slot is reused two logical stages later, the producers often
   spend thousands of cycles waiting for the previous consumer completion.
4. W2 and W3 do not remain phase-aligned. In this pass, the K128 ready delay
   appears mainly as W2 waiting for B0 and W3 waiting for A0.

This supports the performance ablation: reducing TMA transaction and logical
barrier counts did not compensate for the bursty K64/K128 stage cadence.

## Observer effect

This is an actual instrumented execution, not a reconstructed schedule.
However, clock reads and trace state increase the 16K specialization from 166
to 198 registers and add 2592 bytes of static shared memory. Absolute
durations therefore belong to the trace build. Event order, waits, and
dependency relationships are the primary evidence.

## Artifacts

- `alternating_k_pipeline_actual.svg`: plotted pass 3.
- `logs/trace_pass1.csv` through `trace_pass5.csv`: raw trace records.
- `logs/build.txt`: compiler resources.
- `logs/validation.txt`: full-C correctness result.
- `source/gemm_alt_trace.cu`: exact compiled source.
