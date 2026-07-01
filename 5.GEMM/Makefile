NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
GPU_ARCHS ?= -gencode arch=compute_100a,code=sm_100a

TARGET := gemm256_tma_tcgen05_bench
TRACE_TARGET := gemm256_tma_tcgen05_trace
SRC := gemm256_tma_tcgen05_bench.cu
PYTHON ?= python3

SIZES ?= 4096,8192,16384,32768
DEVICE ?= 0
WARMUP ?= 2
ITERS ?= 5
TRACE_SIZE ?= 4096
TRACE_START ?= 56
TRACE_ITERS ?= 8
TRACE_CSV ?= log/gemm256_trace.csv
TRACE_SVG ?= log/gemm256_trace.svg
TRACE_TITLE ?= 5.GEMM pipeline trace size=$(TRACE_SIZE) start=$(TRACE_START) iters=$(TRACE_ITERS)

.PHONY: all build run plot clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(GPU_ARCHS) $< -lcuda -o $@

$(TRACE_TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -DGEMM_CLOCK_TRACE=1 $(GPU_ARCHS) $< -lcuda -o $@

run: build
	./$(TARGET) --device $(DEVICE) --sizes $(SIZES) --warmup $(WARMUP) --iters $(ITERS)

plot: $(TRACE_TARGET)
	mkdir -p $(dir $(TRACE_CSV)) $(dir $(TRACE_SVG))
	./$(TRACE_TARGET) --device $(DEVICE) --sizes $(TRACE_SIZE) --warmup $(WARMUP) --clock-trace --clock-trace-start $(TRACE_START) --clock-trace-iters $(TRACE_ITERS) --trace-csv $(TRACE_CSV)
	$(PYTHON) plot_gemm_trace.py --trace $(TRACE_CSV) --svg $(TRACE_SVG) --title "$(TRACE_TITLE)"

clean:
	rm -f $(TARGET) $(TRACE_TARGET) *.csv
