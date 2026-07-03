CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
PYTHON ?= python3

GENCODE ?= -gencode=arch=compute_100a,code=sm_100a
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) --expt-relaxed-constexpr

SRC ?= main.cu
DEPS ?= attention.cu ptx_wrappers.cuh
BUILD_DIR ?= ../build/0.attention

BLOCKS ?= 4096
# seqlen 1k..32k <-> k_tiles 8..256 (Skv = k_tiles * 128)
SEQLEN ?= 32k
ifeq ($(SEQLEN),1k)
SEQLEN_K_TILES := 8
else ifeq ($(SEQLEN),2k)
SEQLEN_K_TILES := 16
else ifeq ($(SEQLEN),4k)
SEQLEN_K_TILES := 32
else ifeq ($(SEQLEN),8k)
SEQLEN_K_TILES := 64
else ifeq ($(SEQLEN),16k)
SEQLEN_K_TILES := 128
else ifeq ($(SEQLEN),32k)
SEQLEN_K_TILES := 256
else
$(error SEQLEN must be one of 1k/2k/4k/8k/16k/32k (or set K_TILES directly))
endif
K_TILES ?= $(SEQLEN_K_TILES)

# short seqlens: ~0.3ms iters need extra warmup to reach the GPU boost clock
ifeq ($(filter $(SEQLEN),1k 2k 4k),)
WARMUP ?= 3
ITERS ?= 10
else
WARMUP ?= 20
ITERS ?= 50
endif
RUN_CSV ?= log/best.csv

# STABLE=1
STABLE_FLAGS := -DATTENTION_SKIP_V_TMA_EXPECT_TX=0 -DATTENTION_SKIP_V_H0_READY_WAIT=0 -DATTENTION_SKIP_V_H1_READY_WAIT=0
# PERSIST=1 - persistent CONTINUOUS_FLAT scr schedule
PERSIST_FLAGS := -DATTENTION_CONTINUOUS_FLAT=1 $(STABLE_FLAGS) -diag-suppress 177

#   seqlen         1k    2k     4k     8k     16k    32k
#   base (default) 818   1135   1404   1613   1734   1799   (raw ck wobbles)
#   base STABLE=1  789   1110   1381   1586   1714   1780   (all bit-stable)
#   scr            1012  1240   1418   1552   1631   1673   (bit-stable; 1k ~1/12 1ULP)

# Policy: scr <=4k, base >=8k.
BEST_1k  := scr
BEST_2k  := scr
BEST_4k  := scr
BEST_8k  := base
BEST_16k := base
BEST_32k := base

PROFILE_FLAGS :=
PROFILE_SUFFIX :=
ifeq ($(PERSIST),1)
PROFILE_FLAGS := $(PERSIST_FLAGS)
PROFILE_SUFFIX := _scr
endif
ifeq ($(BEST),1)
BEST_PICK := $(BEST_$(SEQLEN))
ifeq ($(BEST_PICK),scr)
PROFILE_FLAGS := $(PERSIST_FLAGS)
endif
PROFILE_SUFFIX := _best_$(SEQLEN)
endif
ifeq ($(STABLE),1)
PROFILE_FLAGS += $(STABLE_FLAGS)
PROFILE_SUFFIX := $(PROFILE_SUFFIX)_stable
endif
# CHECKSUM=1: print raw+masked O_CHECKSUM of the benchmark output after the
# timed run.
ifeq ($(CHECKSUM),1)
PROFILE_FLAGS += -DATTENTION_O_CHECKSUM=1
PROFILE_SUFFIX := $(PROFILE_SUFFIX)_ck
endif
# CAUSAL=1: run the benchmark/validation with the causal mask. Runtime flag
# (same kernel binary, dynamic <0,0,true> dispatch). Supported by the base
# profile and by PERSIST=1 (flat-causal schedule); the causal BEST table is
# not measured yet, so CAUSAL=1 BEST=1 is rejected.
ifeq ($(CAUSAL),1)
ifeq ($(BEST),1)
$(error CAUSAL=1 BEST=1: the BEST table is non-causal; pick base or PERSIST=1 explicitly)
endif
RUN_EXTRA_ARGS := --causal
VALIDATE_EXTRA_ARGS := --causal
PROFILE_SUFFIX := $(PROFILE_SUFFIX)_causal
endif

FASTEST_BIN := $(BUILD_DIR)/attention_custom_kernel$(PROFILE_SUFFIX)
TRACE_BIN := $(BUILD_DIR)/attention_custom_kernel_trace
VALIDATION_BIN := $(BUILD_DIR)/attention_validation$(PROFILE_SUFFIX)

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

.PHONY: build run plot validation validation-suite

build: $(FASTEST_BIN)

$(BUILD_DIR):
	@mkdir -p $@

$(FASTEST_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PROFILE_FLAGS) $(SRC) -o $@ -lcuda

$(VALIDATION_BIN): $(SRC) $(DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PROFILE_FLAGS) $(SRC) -o $@ -lcuda

run: $(FASTEST_BIN)
	$(PYTHON) run.py benchmark --binary $(FASTEST_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(WARMUP) --iters $(ITERS) --csv $(RUN_CSV) $(RUN_EXTRA_ARGS)

plot: | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DATTENTION_CLOCK_TRACE=1 $(SRC) -o $(TRACE_BIN) -lcuda
	$(PYTHON) run.py plot --binary $(TRACE_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(TRACE_WARMUP) --iters $(TRACE_RUN_ITERS) --clock-trace-start $(TRACE_START) --clock-trace-iters $(TRACE_ITERS) --trace-csv $(TRACE_CSV) --summary-csv $(TRACE_SUMMARY_CSV) --svg $(TRACE_SVG)

validation: $(VALIDATION_BIN)
	$(VALIDATION_BIN) --validate --pattern $(VALIDATE_PATTERN) --k-tiles $(VALIDATE_K_TILES) --checksum-repeats $(VALIDATE_CHECKSUM_REPEATS) --csv $(VALIDATE_CSV) $(VALIDATE_EXTRA_ARGS)

# Full fused suite (includes the causal cases in base builds).
validation-suite: $(VALIDATION_BIN)
	$(VALIDATION_BIN) --validate-suite --checksum-repeats $(VALIDATE_CHECKSUM_REPEATS) --csv $(VALIDATE_CSV)
