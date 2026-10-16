.PHONY: all build clean regs sass profile bench

NVCC ?= nvcc
ARCH ?= sm_80
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
BUILD_DIR := build
BIN := $(BUILD_DIR)/tc_gemm
HDRS := runner.cuh $(wildcard kernels/*.cuh)

all: build

build: $(BIN)

$(BUILD_DIR)/runner.o: runner.cu $(HDRS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c runner.cu -o $@

$(BIN): main.cu $(BUILD_DIR)/runner.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) -o $@ main.cu $(BUILD_DIR)/runner.o -lcublas

# 커널별 레지스터·SMEM 사용량 (spill 확인)
regs:
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -c runner.cu -o $(BUILD_DIR)/runner.o 2>&1 \
		| c++filt | grep -E "Function properties|Compiling entry|registers|spill"

# 사용법: make sass KERNEL=tcgemmSwizzle
KERNEL ?= tcgemmSwizzle
sass: build
	cuobjdump -sass $(BIN) | c++filt > $(BUILD_DIR)/all.sass
	@grep -n "Function : .*$(KERNEL)" $(BUILD_DIR)/all.sass | head
	@echo "HMMA count:" $$(grep -c HMMA $(BUILD_DIR)/all.sass)

# 사용법: make profile K=4 SIZE=4096
K ?= 5
SIZE ?= 4096
profile: build
	@mkdir -p results
	ncu --set full -k regex:tcgemm -c 1 --export results/kernel_$(K) --force-overwrite \
		$(BIN) $(K) $(SIZE)

bench: build
	bash bench.sh

clean:
	rm -rf $(BUILD_DIR)

# Native sweep: input generation and FP32 reference run once per matrix size.
$(BUILD_DIR)/tc_tune: tune.cu $(BUILD_DIR)/runner.o $(HDRS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ tune.cu $(BUILD_DIR)/runner.o -lcublas

.PHONY: tune
tune: $(BUILD_DIR)/tc_tune
	$(BUILD_DIR)/tc_tune

$(BUILD_DIR)/tc_check: check.cu $(BUILD_DIR)/runner.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) -o $@ check.cu $(BUILD_DIR)/runner.o -lcublas

.PHONY: check
check: $(BUILD_DIR)/tc_check
	$(BUILD_DIR)/tc_check
