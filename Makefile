
MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CPP_DIR := $(MKFILE_DIR)timemachine/cpp/
INSTALL_PREFIX := $(MKFILE_DIR)timemachine/
PYTEST_CI_ARGS := --color=yes --cov=. --cov-report=html:coverage/ --cov-append --durations=100

NOGPU_MARKER := nogpu
MEMCHECK_MARKER := memcheck
NIGHTLY_MARKER := nightly

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
	pre-commit run --all-files --show-diff-on-failure --color=always

.PHONY: nogpu_tests
nogpu_tests:
	pytest -m $(NOGPU_MARKER) $(PYTEST_CI_ARGS)

.PHONY: memcheck_tests
memcheck_tests:
	compute-sanitizer --launch-timeout 120 --leak-check full --error-exitcode 1 pytest -m $(MEMCHECK_MARKER) $(PYTEST_CI_ARGS)

.PHONY: unit_tests
unit_tests:
	pytest -m 'not $(NOGPU_MARKER) and not $(MEMCHECK_MARKER) and not $(NIGHTLY_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: nightly_tests
nightly_tests:
	pytest -m $(NIGHTLY_MARKER) $(PYTEST_CI_ARGS)

.PHONY: ci
ci: verify memcheck_tests unit_tests
