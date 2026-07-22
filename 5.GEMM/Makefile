NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
EXTRA_NVCCFLAGS ?=
GPU_ARCHS ?= -gencode arch=compute_100a,code=sm_100a

TARGET := gemm256_tma_tcgen05_bench
REPEAT_TARGET := gemm256_tma_tcgen05_repeat
L2_TARGET := gemm256_tma_tcgen05_l2
PERSISTENT_TARGET := gemm256_tma_tcgen05_persistent
PERSISTENT_1777_TARGET := gemm256_tma_tcgen05_persistent_1777
REPEAT_128X256_TARGET := gemm128x256_tma_tcgen05_repeat_broadcast
TRACE_TARGET := gemm256_tma_tcgen05_trace
SRC := gemm256_tma_tcgen05_bench.cu
PYTHON ?= python3

SIZES ?= 4096,8192,16384,32768
DEVICE ?= 0
WARMUP ?= 1
ITERS ?= 5
TRACE_SIZE ?= 4096
TRACE_START ?= 56
TRACE_ITERS ?= 8
TRACE_CSV ?= log/gemm256_trace.csv
TRACE_SVG ?= log/gemm256_trace.svg
TRACE_TITLE ?= 5.GEMM pipeline trace size=$(TRACE_SIZE) start=$(TRACE_START) iters=$(TRACE_ITERS)

# Default persistent configuration: the selected dense-address L2 schedule.
PERSISTENT_CONFIG_FLAGS := \
	-DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1 \
	-DGEMM_PERSISTENT_CTA=1 \
	-DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16 \
	-DGEMM_PERSISTENT_8K_MACRO_M=12 -DGEMM_PERSISTENT_8K_MACRO_N=12 \
	-DGEMM_PERSISTENT_32K_MACRO_M=12 -DGEMM_PERSISTENT_32K_MACRO_N=12 \
	-DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1

# Historical 1777-series scheduler.  Keep this independent from the selected
# default so later L2 tuning cannot silently change the frozen comparison.
PERSISTENT_1777_CONFIG_FLAGS := \
	-DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1 \
	-DGEMM_PERSISTENT_CTA=1 \
	-DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16 \
	-DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16 \
	-DGEMM_PERSISTENT_32K_MACRO_M=8 -DGEMM_PERSISTENT_32K_MACRO_N=18 \
	-DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1

REPEAT_128X256_FLAGS := \
	-DGEMM_REPEAT_INPUT=1 -DGEMM_REPEAT_TUNING=1 \
	-DGEMM_PERSISTENT_CTA=1 \
	-DGEMM_CTA_M=128 -DGEMM_STAGE_K=128 -DGEMM_STAGES=3 \
	-DGEMM_REPEAT_B_BROADCAST=1 \
	-DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16 \
	-DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16 \
	-DGEMM_PERSISTENT_32K_MACRO_M=8 -DGEMM_PERSISTENT_32K_MACRO_N=18 \
	-DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1

.PHONY: all build build-repeat build-repeat-128x256 build-l2 build-persistent build-persistent-1777 run run-repeat run-repeat-128x256 run-l2 run-persistent run-persistent-1777 plot clean

all: build-persistent

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(EXTRA_NVCCFLAGS) $(GPU_ARCHS) $< -lcuda -o $@

build-repeat: $(REPEAT_TARGET)

$(REPEAT_TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(EXTRA_NVCCFLAGS) -DGEMM_REPEAT_INPUT=1 $(GPU_ARCHS) $< -lcuda -o $@

build-repeat-128x256: $(REPEAT_128X256_TARGET)

$(REPEAT_128X256_TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(EXTRA_NVCCFLAGS) $(REPEAT_128X256_FLAGS) $(GPU_ARCHS) $< -lcuda -o $@

build-l2: $(L2_TARGET)

$(L2_TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(EXTRA_NVCCFLAGS) -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1 $(GPU_ARCHS) $< -lcuda -o $@

$(PERSISTENT_TARGET): $(SRC) Makefile
	$(NVCC) $(CXXFLAGS) $(EXTRA_NVCCFLAGS) $(PERSISTENT_CONFIG_FLAGS) $(GPU_ARCHS) $< -lcuda -o $@

build-persistent: $(PERSISTENT_TARGET)

# Freeze the scheduler configuration that produced 1777.128 TFLOP/s at 16K
# on Vast.ai instance 45378714.  Keep this target separate from subsequent L2
# tuning so future default changes do not silently alter the paper baseline.
$(PERSISTENT_1777_TARGET): $(SRC) Makefile
	$(NVCC) $(CXXFLAGS) $(EXTRA_NVCCFLAGS) $(PERSISTENT_1777_CONFIG_FLAGS) $(GPU_ARCHS) $< -lcuda -o $@

build-persistent-1777: $(PERSISTENT_1777_TARGET)

$(TRACE_TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(EXTRA_NVCCFLAGS) -DGEMM_CLOCK_TRACE=1 $(GPU_ARCHS) $< -lcuda -o $@

run: $(PERSISTENT_TARGET)
	./$(PERSISTENT_TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS) --input-init random --persistent-ctas $(PERSISTENT_CTAS) --csv gemm256_tma_tcgen05_persistent.csv

run-repeat: build-repeat
	./$(REPEAT_TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS) --input-init random --csv gemm256_tma_tcgen05_repeat.csv

run-repeat-128x256: build-repeat-128x256
	./$(REPEAT_128X256_TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS) --input-init random --persistent-ctas $(PERSISTENT_CTAS) --csv gemm128x256_tma_tcgen05_repeat_broadcast.csv

run-l2: build-l2
	./$(L2_TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS) --input-init random --csv gemm256_tma_tcgen05_l2.csv

PERSISTENT_CTAS ?= 148
run-persistent: $(PERSISTENT_TARGET)
	./$(PERSISTENT_TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS) --input-init random --persistent-ctas $(PERSISTENT_CTAS) --csv gemm256_tma_tcgen05_persistent.csv

run-persistent-1777: $(PERSISTENT_1777_TARGET)
	./$(PERSISTENT_1777_TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS) --input-init random --persistent-ctas $(PERSISTENT_CTAS) --csv gemm256_tma_tcgen05_persistent_1777.csv

plot: $(TRACE_TARGET)
	mkdir -p $(dir $(TRACE_CSV)) $(dir $(TRACE_SVG))
	./$(TRACE_TARGET) --device $(DEVICE) --sizes $(TRACE_SIZE) --warmup $(WARMUP) --clock-trace --clock-trace-start $(TRACE_START) --clock-trace-iters $(TRACE_ITERS) --trace-csv $(TRACE_CSV)
	$(PYTHON) plot_gemm_trace.py --trace $(TRACE_CSV) --svg $(TRACE_SVG) --title "$(TRACE_TITLE)"

clean:
	rm -f $(TARGET) $(REPEAT_TARGET) $(REPEAT_128X256_TARGET) $(L2_TARGET) $(PERSISTENT_TARGET) $(PERSISTENT_1777_TARGET) $(TRACE_TARGET) *.csv
