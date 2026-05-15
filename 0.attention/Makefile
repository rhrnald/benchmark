CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

GENCODE ?= -gencode=arch=compute_100a,code=sm_100a
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) --expt-relaxed-constexpr

SRC ?= main.cu
BUILD_DIR ?= ../build/0.attention
FASTEST_BIN ?= $(BUILD_DIR)/attention_custom_kernel
VALIDATION_BIN ?= $(BUILD_DIR)/attention_validation

BLOCKS ?= 4096
K_TILES ?= 256
WARMUP ?= 3
ITERS ?= 10
CSV ?= /tmp/attention_main.csv

VALIDATE_PATTERN ?= rank1
VALIDATE_K_TILES ?= 4
VALIDATE_CSV ?= /tmp/attention_main_validate_rank1_k4.csv

.DEFAULT_GOAL := run

.PHONY: build run validation

build: $(FASTEST_BIN)

$(BUILD_DIR):
	@mkdir -p $@

$(FASTEST_BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ -lcuda

$(VALIDATION_BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DATTENTION_STORE_OUTPUT=1 $(SRC) -o $@ -lcuda

run: $(FASTEST_BIN)
	$(FASTEST_BIN) --blocks $(BLOCKS) --k-tiles $(K_TILES) --warmup $(WARMUP) --iters $(ITERS) --csv $(CSV)

validation: $(VALIDATION_BIN)
	$(VALIDATION_BIN) --validate --pattern $(VALIDATE_PATTERN) --k-tiles $(VALIDATE_K_TILES) --csv $(VALIDATE_CSV)
