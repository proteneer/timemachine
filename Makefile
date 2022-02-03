
MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CPP_DIR := $(MKFILE_DIR)timemachine/cpp/
INSTALL_PREFIX := $(MKFILE_DIR)timemachine/
PYTEST_CI_ARGS := --cov=. --cov-report=term-missing --durations=100

NPROCS = `nproc`

.PHONY: build
build:
	mkdir -p $(CPP_DIR)build/ && cd $(CPP_DIR)build/ &&  \
	cmake -DCUDA_ARCH=$(shell echo $(CUDA_ARCH) | sed -e 's/^sm_//') -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) ../ && \
	make -j$(NPROCS) install

clean:
	cd $(CPP_DIR) && rm -rf build/ | true

ci:
	pre-commit run --all-files --show-diff-on-failure && \
	export PYTHONPATH=$(MKFILE_DIR) && \
	cuda-memcheck --leak-check full --error-exitcode 1 pytest $(PYTEST_CI_ARGS) tests/ && \
	pytest $(PYTEST_CI_ARGS) slow_tests/
