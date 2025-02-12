
MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CPP_DIR := $(MKFILE_DIR)timemachine/cpp/
INSTALL_PREFIX := $(MKFILE_DIR)timemachine/
# Conditionally set pytest args, to be able to override in CI
PYTEST_CI_ARGS ?= --color=yes --cov=. --cov-report=html:coverage/ --cov-append --durations=100 -n auto

# pytest mark to indicate tests that should be run in an environment without CUDA.
# (e.g. no nvcc, so we can't build custom_ops)
NOCUDA_MARKER := nocuda

# pytest mark to indicate tests that should be run in an environment without a GPU
# (but WITH build dependencies of # custom_ops, e.g. CUDA).
# These tests can be run on cheaper CPU instances.
NOGPU_MARKER := nogpu

MEMCHECK_MARKER := memcheck
NIGHTLY_MARKER := nightly

COMPUTE_SANITIZER_CMD := compute-sanitizer --launch-timeout 120 --padding 2048 --tool memcheck --leak-check full --error-exitcode 1

NPROCS = `nproc`

.PHONY: build
build:
	mkdir -p $(CPP_DIR)build/ && cd $(CPP_DIR)build/ &&  \
	cmake -DCUDA_ARCH=$(CUDA_ARCH) -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) ../ && \
	make -j$(NPROCS) install

clean:
	cd $(CPP_DIR) && rm -rf build/ | true

.PHONY: verify
verify:
	pre-commit run --all-files --show-diff-on-failure --color=always

.PHONY: nocuda_tests
nocuda_tests:
	pytest -m '$(NOCUDA_MARKER) and not $(NIGHTLY_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: nogpu_tests
nogpu_tests:
	pytest -m '$(NOGPU_MARKER) and not $(NIGHTLY_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: memcheck_tests
memcheck_tests:
	$(COMPUTE_SANITIZER_CMD) pytest -m '$(MEMCHECK_MARKER) and not $(NIGHTLY_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: unit_tests
unit_tests:
	pytest -m 'not $(NOCUDA_MARKER) and not $(NOGPU_MARKER) and not $(MEMCHECK_MARKER) and not $(NIGHTLY_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: nightly_tests
nightly_tests:
	pytest -m '$(NIGHTLY_MARKER) and not $(NOCUDA_MARKER) and not $(NOGPU_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: nightly_memcheck_tests
nightly_memcheck_tests:
	$(COMPUTE_SANITIZER_CMD) pytest -m '$(NIGHTLY_MARKER) and $(MEMCHECK_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: nocuda_nightly_tests
nocuda_nightly_tests:
	pytest -m '$(NIGHTLY_MARKER) and $(NOCUDA_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: nogpu_nightly_tests
nogpu_nightly_tests:
	pytest -m '$(NIGHTLY_MARKER) and $(NOGPU_MARKER)' $(PYTEST_CI_ARGS)

.PHONY: ci
ci: verify memcheck_tests unit_tests
