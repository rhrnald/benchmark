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
EARLY_COMMIT_BIN ?= $(BUILD_DIR)/attention_early_commit_measure

BLOCKS ?= 4096
K_TILES ?= 256
WARMUP ?= 3
ITERS ?= 10
RUN_CSV ?= log/best.csv

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

EARLY_STARTS ?= 30,40,50,60
EARLY_SAMPLES ?= 100
EARLY_OUT_DIR ?= log/early_commit_measure

.DEFAULT_GOAL := build

.PHONY: build run plot validation early-commit-measure

build: $(FASTEST_BIN)

$(BUILD_DIR):
	@mkdir -p $@

$(FASTEST_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ -lcuda

$(TRACE_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DATTENTION_CLOCK_TRACE=1 $(SRC) -o $@ -lcuda

$(VALIDATION_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ -lcuda

$(EARLY_COMMIT_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DATTENTION_CLOCK_TRACE=1 -DATTENTION_EARLY_COMMIT_MEASURE=1 $(SRC) -o $@ -lcuda

run: $(FASTEST_BIN)
	$(PYTHON) run.py benchmark --binary $(FASTEST_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(WARMUP) --iters $(ITERS) --csv $(RUN_CSV)

plot: $(TRACE_BIN)
	$(PYTHON) run.py plot --binary $(TRACE_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(TRACE_WARMUP) --iters $(TRACE_RUN_ITERS) --clock-trace-start $(TRACE_START) --clock-trace-iters $(TRACE_ITERS) --trace-csv $(TRACE_CSV) --summary-csv $(TRACE_SUMMARY_CSV) --svg $(TRACE_SVG)

validation: $(VALIDATION_BIN)
	$(VALIDATION_BIN) --validate --pattern $(VALIDATE_PATTERN) --k-tiles $(VALIDATE_K_TILES) --checksum-repeats $(VALIDATE_CHECKSUM_REPEATS) --csv $(VALIDATE_CSV)

early-commit-measure: $(EARLY_COMMIT_BIN)
	$(PYTHON) run.py early-commit-measure --binary $(EARLY_COMMIT_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(WARMUP) --starts $(EARLY_STARTS) --samples $(EARLY_SAMPLES) --out-dir $(EARLY_OUT_DIR)
