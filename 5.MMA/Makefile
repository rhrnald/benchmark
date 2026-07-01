NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
GPU_ARCHS ?= -gencode arch=compute_100a,code=sm_100a

TARGET := gemm256_tma_tcgen05_bench
SRC := gemm256_tma_tcgen05_bench.cu

SIZES ?= 4096,8192,16384,32768
DEVICE ?= 0
WARMUP ?= 2
ITERS ?= 5

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(GPU_ARCHS) $< -lcuda -o $@

run: build
	./$(TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS)

clean:
	rm -f $(TARGET) *.csv
