NVCC ?= nvcc
ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda
HEADERS := $(wildcard kernels/*.cuh) runner.cuh
.PHONY: all check bench best tune regs profile clean
all: build/bf16_gemm
build/runner.o: runner.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -Xptxas=-v -c $< -o $@ 2>build/resources.log
build/bf16_gemm: main.cu build/runner.o runner.cuh
	$(NVCC) $(NVCCFLAGS) $< build/runner.o -lcublas -lcublasLt -o $@
check: all
	python3 check_layout.py
	python3 -m unittest discover -s . -p test_analysis.py
	./build/bf16_gemm check
bench: all
	./build/bf16_gemm bench
best: all
	python3 benchmark_selected.py
tune: all
	./build/bf16_gemm tune 4096
regs: all
	cat build/resources.log
K ?= q128_s2
SIZE ?= 4096
GROUP ?= 8
profile: all
	@mkdir -p results
	ncu --profile-from-start off --clock-control none --section SpeedOfLight --section LaunchStats --section SourceCounters --export results/profile-$(K)-$(SIZE) --force-overwrite ./build/bf16_gemm profile $(K) $(SIZE) $(GROUP)
clean:
	rm -rf build
