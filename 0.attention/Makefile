CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
PYTHON ?= python3

GENCODE ?= -gencode=arch=compute_100a,code=sm_100a
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) --expt-relaxed-constexpr

SRC ?= main.cu
DEPS ?= attention.cu ptx_wrappers.cuh
BUILD_DIR ?= ../build/0.attention
FASTEST_BIN ?= $(BUILD_DIR)/attention_custom_kernel
TRACE_BIN ?= $(BUILD_DIR)/attention_custom_kernel_trace
VALIDATION_BIN ?= $(BUILD_DIR)/attention_validation

BLOCKS ?= 4096
# support seqlen 1k, 32k
SEQLEN ?= 32k
ifeq ($(SEQLEN),1k)
SEQLEN_K_TILES := 8
else ifeq ($(SEQLEN),32k)
SEQLEN_K_TILES := 256
else
$(error SEQLEN must be 1k or 32k (or set K_TILES directly))
endif
K_TILES ?= $(SEQLEN_K_TILES)
# 1k iters are ~0.3ms each, need more warmup to reach the GPU boost clock
ifeq ($(SEQLEN),1k)
WARMUP ?= 20
ITERS ?= 50
else
WARMUP ?= 3
ITERS ?= 10
endif
RUN_CSV ?= log/best.csv

# persistent occupancy-1 kernel with cross-tile QK peel: ~963 TFLOPS at
ifeq ($(FAST),1)
NVCCFLAGS += -DATTENTION_PERSISTENT=1 -DATTENTION_PERSISTENT_OVERLAP=1 -DATTENTION_PERSISTENT_OVERLAP_EARLY=1 -DATTENTION_PERSISTENT_OVERLAP_O_IN_V=1 -DATTENTION_PERSISTENT_OVERLAP_DEFER_STORE=1 -DATTENTION_PERSISTENT_OVERLAP_QK_PEEL=1 -DATTENTION_PEEL_PREARM=1 -DATTENTION_PEEL_DURING_DRAIN=1 -DATTENTION_PEEL_AFTER_SYNC=1 -diag-suppress 177
FASTEST_BIN := $(BUILD_DIR)/attention_custom_kernel_fast
endif

TRACE_START ?= 56
TRACE_ITERS ?= 8
TRACE_RUN_ITERS ?= 1
TRACE_WARMUP ?= 0
TRACE_CSV ?= log/best_trace.csv
TRACE_SUMMARY_CSV ?= log/best_trace_summary.csv
TRACE_SVG ?= log/best.svg

VALIDATE_PATTERN ?= rank1
VALIDATE_K_TILES ?= 4
VALIDATE_CHECKSUM_REPEATS ?= 2
VALIDATE_CSV ?= /tmp/attention_main_validate_rank1_k4.csv

.DEFAULT_GOAL := build

.PHONY: build run plot validation

build: $(FASTEST_BIN)

$(BUILD_DIR):
	@mkdir -p $@

$(FASTEST_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ -lcuda

$(TRACE_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DATTENTION_CLOCK_TRACE=1 $(SRC) -o $@ -lcuda

$(VALIDATION_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ -lcuda

run: $(FASTEST_BIN)
	$(PYTHON) run.py benchmark --binary $(FASTEST_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(WARMUP) --iters $(ITERS) --csv $(RUN_CSV)

plot: $(TRACE_BIN)
	$(PYTHON) run.py plot --binary $(TRACE_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(TRACE_WARMUP) --iters $(TRACE_RUN_ITERS) --clock-trace-start $(TRACE_START) --clock-trace-iters $(TRACE_ITERS) --trace-csv $(TRACE_CSV) --summary-csv $(TRACE_SUMMARY_CSV) --svg $(TRACE_SVG)

validation: $(VALIDATION_BIN)
	$(VALIDATION_BIN) --validate --pattern $(VALIDATE_PATTERN) --k-tiles $(VALIDATE_K_TILES) --checksum-repeats $(VALIDATE_CHECKSUM_REPEATS) --csv $(VALIDATE_CSV)
