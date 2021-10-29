
MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CPP_DIR := $(MKFILE_DIR)timemachine/cpp/
INSTALL_PREFIX := $(MKFILE_DIR)timemachine/
PYTEST_CI_ARGS := --cov=. --cov-report=term-missing
BLACK_FLAGS := --line-length 120 .

NPROCS = `nproc`

CUDA_ARCH := "sm_70"

.PHONY: build
build:
	mkdir -p $(CPP_DIR)build/ && cd $(CPP_DIR)build/ &&  \
	cmake -DCUDA_ARCH=$(CUDA_ARCH) -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) ../ && \
	make -j$(NPROCS) install

clean:
	cd $(CPP_DIR) && rm -rf build/ | true

fmt:
	black $(BLACK_FLAGS)

ci:
	black --check $(BLACK_FLAGS) && \
	export PYTHONPATH=$(MKFILE_DIR) && \
	cuda-memcheck pytest $(PYTEST_CI_ARGS) tests/ && \
	pytest $(PYTEST_CI_ARGS) slow_tests/