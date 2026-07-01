NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
GPU_ARCHS ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_100,code=sm_100

TARGET := cublas_gemm_bench
SRC := cublas_gemm_bench.cu

DEVICE ?= 0
M ?= 16384
N ?= 16384
K ?= 16384
REPEAT ?= 50
WARMUP ?= 10
MODES ?= fp16 bf16 tf32

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(GPU_ARCHS) $< -lcublas -o $@

run: build
	@for mode in $(MODES); do \
		./$(TARGET) --device $(DEVICE) --m $(M) --n $(N) --k $(K) \
			--repeat $(REPEAT) --warmup $(WARMUP) --mode $$mode; \
	done

clean:
	rm -f $(TARGET)
