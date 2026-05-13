CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

GENCODE ?= -gencode=arch=compute_100a,code=sm_100a
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) --expt-relaxed-constexpr
CUTLASS_INCLUDE ?= -Ikernel_candidates/repos/cutlass/include
BUILD_DIR ?= build
ATTENTION_SRC ?= 0.attention/attention_fused_real_attention.cu
ATTENTION_COMPARE_CSV ?= 0.attention/attention_compare_default_b1_h16_s32k.csv
ATTENTION_COMPARE_BLOCKS ?= 4096
ATTENTION_COMPARE_K_TILES ?= 256
ATTENTION_COMPARE_WARMUP ?= 3
ATTENTION_COMPARE_ITERS ?= 10

.PHONY: all clean run attention_custom_kernel attention_custom_kernel_fastest attention_custom_kernel_store_output attention_actual_kernel attention_validation attention_validation_fastest attention_validation_actual attention_compare_default tcgen05_ld_pack_toy tcgen05_ld_mma_overlap_toy tma_smem_store_overlap_toy mma_throughput_bench tmem_ldst_bench mma_ld_pipeline_bench mma_ld_pipeline_128kb_case tma_mma_ld_pipeline_bench tma_multicast_bench

all: attention_custom_kernel tmem_ldst_bench blackwell_pipeline_overlap_bench mma_throughput_bench mma_ld_pipeline_bench mma_ld_pipeline_128kb_case tma_mma_ld_pipeline_bench tma_multicast_bench tmem_mma_overlap_bench tcgen05_alloc_occupancy_probe mma_four_warpgroups_consume_bench tcgen05_fabric_sharing_bench qk_two_producer_consume_pipeline tmem_scale_accumulate_bench tma_smem_store_overlap_toy

attention_custom_kernel:
	@mkdir -p $(BUILD_DIR)/0.attention
	$(NVCC) $(NVCCFLAGS) $(ATTENTION_SRC) -o $(BUILD_DIR)/0.attention/attention_custom_kernel -lcuda

attention_custom_kernel_fastest:
	@mkdir -p $(BUILD_DIR)/0.attention
	$(NVCC) $(NVCCFLAGS) $(ATTENTION_SRC) -o $(BUILD_DIR)/0.attention/attention_custom_kernel -lcuda

attention_custom_kernel_store_output:
	@mkdir -p $(BUILD_DIR)/0.attention
	$(NVCC) $(NVCCFLAGS) -DATTENTION_STORE_OUTPUT=1 $(ATTENTION_SRC) -o $(BUILD_DIR)/0.attention/attention_custom_kernel_store_output -lcuda

attention_actual_kernel:
	@mkdir -p $(BUILD_DIR)/0.attention
	$(NVCC) $(NVCCFLAGS) -DATTENTION_STORE_OUTPUT=1 $(ATTENTION_SRC) -o $(BUILD_DIR)/0.attention/attention_actual_kernel -lcuda

attention_validation:
	@mkdir -p $(BUILD_DIR)/0.attention
	$(NVCC) $(NVCCFLAGS) -DATTENTION_STORE_OUTPUT=1 $(ATTENTION_SRC) -o $(BUILD_DIR)/0.attention/attention_validation -lcuda

attention_validation_fastest:
	@mkdir -p $(BUILD_DIR)/0.attention
	$(NVCC) $(NVCCFLAGS) -DATTENTION_STORE_OUTPUT=1 $(ATTENTION_SRC) -o $(BUILD_DIR)/0.attention/attention_validation -lcuda

attention_validation_actual:
	@mkdir -p $(BUILD_DIR)/0.attention
	$(NVCC) $(NVCCFLAGS) -DATTENTION_STORE_OUTPUT=1 $(ATTENTION_SRC) -o $(BUILD_DIR)/0.attention/attention_validation_actual -lcuda

attention_compare_default: attention_custom_kernel
	$(BUILD_DIR)/0.attention/attention_custom_kernel --blocks $(ATTENTION_COMPARE_BLOCKS) --k-tiles $(ATTENTION_COMPARE_K_TILES) --warmup $(ATTENTION_COMPARE_WARMUP) --iters $(ATTENTION_COMPARE_ITERS) --csv $(ATTENTION_COMPARE_CSV)

tcgen05_ld_pack_toy:
	@mkdir -p $(BUILD_DIR)/0-1.TCGEN05_LD_PACK_TOY
	$(NVCC) $(NVCCFLAGS) 0-1.TCGEN05_LD_PACK_TOY/tcgen05_ld_pack_toy.cu -o $(BUILD_DIR)/0-1.TCGEN05_LD_PACK_TOY/tcgen05_ld_pack_toy

tcgen05_ld_mma_overlap_toy:
	@mkdir -p $(BUILD_DIR)/0-2.TCGEN05_LD_MMA_OVERLAP_TOY
	$(NVCC) $(NVCCFLAGS) 0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_overlap_toy.cu -o $(BUILD_DIR)/0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_overlap_toy

tma_smem_store_overlap_toy:
	@mkdir -p $(BUILD_DIR)/0-3.TMA_SMEM_STORE_OVERLAP_TOY
	$(NVCC) $(NVCCFLAGS) 0-3.TMA_SMEM_STORE_OVERLAP_TOY/tma_smem_store_overlap_toy.cu -o $(BUILD_DIR)/0-3.TMA_SMEM_STORE_OVERLAP_TOY/tma_smem_store_overlap_toy -lcuda

tmem_ldst_bench:
	@mkdir -p $(BUILD_DIR)/2.TMEM_LDST
	$(NVCC) $(NVCCFLAGS) 2.TMEM_LDST/tmem_ldst_bench.cu -o $(BUILD_DIR)/2.TMEM_LDST/tmem_ldst_bench

blackwell_pipeline_overlap_bench: blackwell_pipeline_overlap_bench.cu
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_INCLUDE) $< -o $@

mma_throughput_bench:
	@mkdir -p $(BUILD_DIR)/1.mma
	$(NVCC) $(NVCCFLAGS) 1.mma/mma_throughput_bench.cu -o $(BUILD_DIR)/1.mma/mma_throughput_bench

mma_ld_pipeline_bench:
	@mkdir -p $(BUILD_DIR)/3.MMA_LD_PIPELINE
	$(NVCC) $(NVCCFLAGS) 3.MMA_LD_PIPELINE/mma_ld_pipeline_bench.cu -o $(BUILD_DIR)/3.MMA_LD_PIPELINE/mma_ld_pipeline_bench

mma_ld_pipeline_128kb_case:
	@mkdir -p $(BUILD_DIR)/3-1.MMA_LD_PIPELINE_128KB
	$(NVCC) $(NVCCFLAGS) 3-1.MMA_LD_PIPELINE_128KB/mma_ld_pipeline_128kb_case.cu -o $(BUILD_DIR)/3-1.MMA_LD_PIPELINE_128KB/mma_ld_pipeline_128kb_case

tma_mma_ld_pipeline_bench:
	@mkdir -p $(BUILD_DIR)/3-2.TMA_MMA_LD_PIPELINE
	$(NVCC) $(NVCCFLAGS) 3-2.TMA_MMA_LD_PIPELINE/tma_mma_ld_pipeline_bench.cu -o $(BUILD_DIR)/3-2.TMA_MMA_LD_PIPELINE/tma_mma_ld_pipeline_bench -lcuda

tma_multicast_bench:
	@mkdir -p $(BUILD_DIR)/4.TMA_MULTICAST
	$(NVCC) $(NVCCFLAGS) 4.TMA_MULTICAST/tma_multicast_bench.cu -o $(BUILD_DIR)/4.TMA_MULTICAST/tma_multicast_bench -lcuda

tmem_mma_overlap_bench: tmem_mma_overlap_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

mma_four_warpgroups_consume_bench: mma_four_warpgroups_consume_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

tcgen05_fabric_sharing_bench: tcgen05_fabric_sharing_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

qk_two_producer_consume_pipeline: qk_two_producer_consume_pipeline.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcuda

tmem_scale_accumulate_bench: tmem_scale_accumulate_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

tcgen05_alloc_occupancy_probe: tcgen05_alloc_occupancy_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: tmem_ldst_bench
	$(BUILD_DIR)/2.TMEM_LDST/tmem_ldst_bench --csv 2.TMEM_LDST/tmem_ldst_bench.csv

clean:
	rm -f tmem_ldst_bench blackwell_pipeline_overlap_bench mma_throughput_bench tmem_mma_overlap_bench tcgen05_alloc_occupancy_probe mma_four_warpgroups_consume_bench tcgen05_fabric_sharing_bench qk_two_producer_consume_pipeline tmem_scale_accumulate_bench
	rm -rf $(BUILD_DIR)/0.attention $(BUILD_DIR)/0-1.TCGEN05_LD_PACK_TOY $(BUILD_DIR)/0-2.TCGEN05_LD_MMA_OVERLAP_TOY $(BUILD_DIR)/0-3.TMA_SMEM_STORE_OVERLAP_TOY $(BUILD_DIR)/1.mma $(BUILD_DIR)/2.TMEM_LDST $(BUILD_DIR)/3.MMA_LD_PIPELINE $(BUILD_DIR)/3-1.MMA_LD_PIPELINE_128KB $(BUILD_DIR)/3-2.TMA_MMA_LD_PIPELINE $(BUILD_DIR)/4.TMA_MULTICAST
