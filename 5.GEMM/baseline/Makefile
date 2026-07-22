CUDA_HOME ?= /usr/local/cuda-12.9
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump

TARGET ?= gemm256_bf16_16k
SOURCE := gemm256_bf16_16k.cu
NVCCFLAGS ?= -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
LDLIBS ?= -lcuda

WARMUP ?= 1
ITERS ?= 5
INPUT ?= random
CSV ?= gemm256_bf16_16k.csv

.PHONY: all build run run-signed8 validate resources clean

all: build

build: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $< $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET) --warmup $(WARMUP) --iters $(ITERS) \
		--input-init $(INPUT) \
		--csv $(CSV)

run-signed8: $(TARGET)
	./$(TARGET) --warmup $(WARMUP) --iters $(ITERS) \
		--input-init random-signed8 \
		--csv $(CSV)

validate: $(TARGET)
	./$(TARGET) --validate --validate-size 512 \
		--validate-pattern pattern

resources: $(TARGET)
	$(CUOBJDUMP) --dump-resource-usage $(TARGET)

clean:
	rm -f -- $(TARGET)
