
MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CPP_DIR := $(MKFILE_DIR)timemachine/cpp/
INSTALL_PREFIX := $(MKFILE_DIR)timemachine/
PYTEST_CI_ARGS := --cov=. --cov-report=html:coverage/ --cov-append --durations=100

MEMCHECK_MARKER := memcheck

NPROCS = `nproc`

.PHONY: build
build:
	mkdir -p $(CPP_DIR)build/ && cd $(CPP_DIR)build/ &&  \
	cmake -DCUDA_ARCH=$(shell echo $(CUDA_ARCH) | sed -e 's/^sm_//') -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) ../ && \
	make -j$(NPROCS) install

clean:
	cd $(CPP_DIR) && rm -rf build/ | true

.PHONY: grpc
grpc:
	python -m grpc_tools.protoc -I grpc/ --python_out=. --grpc_python_out=. grpc/timemachine/parallel/grpc/service.proto

.PHONY: verify
verify:
	pre-commit run --all-files --show-diff-on-failure

.PHONY: memcheck_tests
memcheck_tests:
	cuda-memcheck --leak-check full --error-exitcode 1 pytest -m $(MEMCHECK_MARKER) $(PYTEST_CI_ARGS)

.PHONY: unit_tests
unit_tests:
	pytest -m 'not $(MEMCHECK_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: ci
ci: verify memcheck_tests unit_tests
