CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

GENCODE ?= -gencode=arch=compute_100a,code=sm_100a
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) --expt-relaxed-constexpr

BLOCKS ?= 512
TARGET_MMAS ?= 8
FULL_EXTRA_MMAS ?= 0
WARPS ?= 4
BUILD_DIR ?= ../build/early_commit
BIN ?= $(BUILD_DIR)/early_commit_race_t$(TARGET_MMAS)_f$(FULL_EXTRA_MMAS)_w$(WARPS)
OUT_DIR ?= log
EARLY_TARGETS ?= 8
EARLY_EXTRAS ?= 0
DELAYS ?= 0,1,2,3,4,5,6,7,8,10,12,16,24,32,48,64,96,128
WARMUP ?= 1
PROBE_GAP_CYCLES ?= 2048
COMPILE_DEFS := -DEARLY_COMMIT_TARGET_MMAS=$(TARGET_MMAS) \
	-DEARLY_COMMIT_FULL_EXTRA_MMAS=$(FULL_EXTRA_MMAS) \
	-DEARLY_COMMIT_WARPS=$(WARPS)

DETAIL_CSV ?= $(OUT_DIR)/early_commit_race_detail.csv
SUMMARY_CSV ?= $(OUT_DIR)/early_commit_race_summary.csv
MODEL_CSV ?= $(OUT_DIR)/early_commit_model.csv

.PHONY: build run model smoke clean

build: $(BIN)

$(BUILD_DIR):
	mkdir -p $@

$(BIN): early_commit_race.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(COMPILE_DEFS) $< -o $@

$(OUT_DIR):
	mkdir -p $@

run: $(BIN) | $(OUT_DIR)
	$(BIN) --blocks $(BLOCKS) --target-mmas $(TARGET_MMAS) \
		--full-extra-mmas $(FULL_EXTRA_MMAS) --early-targets $(EARLY_TARGETS) \
		--early-extras $(EARLY_EXTRAS) --delays $(DELAYS) --warmup $(WARMUP) \
		--csv $(DETAIL_CSV) --summary-csv $(SUMMARY_CSV)

model: $(BIN) | $(OUT_DIR)
	$(BIN) --model --blocks $(BLOCKS) --target-mmas $(TARGET_MMAS) \
		--full-extra-mmas $(FULL_EXTRA_MMAS) --early-targets $(EARLY_TARGETS) \
		--warmup $(WARMUP) --probe-gap-cycles $(PROBE_GAP_CYCLES) \
		--model-csv $(MODEL_CSV)

smoke: $(BIN) | $(OUT_DIR)
	$(BIN) --blocks 32 --target-mmas $(TARGET_MMAS) --full-extra-mmas $(FULL_EXTRA_MMAS) \
		--early-targets 7,8 --early-extras 0 --delays 0,128,256 \
		--warmup 0 --csv $(OUT_DIR)/smoke_detail.csv \
		--summary-csv $(OUT_DIR)/smoke_summary.csv

clean:
	rm -rf $(BUILD_DIR) $(OUT_DIR)
