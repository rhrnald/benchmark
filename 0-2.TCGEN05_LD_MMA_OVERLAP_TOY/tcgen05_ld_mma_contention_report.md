# tcgen05.ld vs tcgen05.mma Contention

## Reproduction

```bash
cd /home/snu_avq1/workspace/chaewon/benchmark
make tcgen05_ld_mma_overlap_toy
./build/0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_overlap_toy \
  --peak-only --blocks 4096 --repeats 8192 --ld-repeats 81920 \
  --warmup 2 --iters 5
python3 0-2.TCGEN05_LD_MMA_OVERLAP_TOY/analyze_ld_mma_contention.py
```

## Result

- LD-only peak: `692.673 TB/s`
- LD while MMA overlaps: `410.096 TB/s`
- Observed LD drop: `282.576 TB/s` = `1653.1 B/cyc/SM`
- Reference MMA peak: `2207.606 TFLOP/s`
- Acc32 C tile size: `65536 B`
- Reference MMA C-read demand: `275.951 TB/s` = `1614.3 B/cyc/SM`
- Ratio, observed drop / C-read demand: `1.024`

## Claim

The observed loss in tcgen05.ld bandwidth is approximately equal to the C-read bandwidth demanded by peak tcgen05.mma. This supports the claim that tcgen05.mma accumulator C-side traffic competes with tcgen05.ld for the same TMEM read/fabric resource.
