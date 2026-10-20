-include .env

# an image name, with or without a :tag
BUILDER_IMAGE  ?= vllm37-builder
# an image name, with or without a :tag
RUNTIME_LOCAL  ?= vllm37-local
# a whole number from 1
JOBS           ?= 4
REPO_ROOT      := $(shell git rev-parse --show-toplevel 2>/dev/null || echo ../..)

.PHONY: help build-builder build-local build-local-no-cache build-all-local run stop logs clean verify-cutlass-patches verify-xformers-patches

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build-builder: ## Build the builder image (CUDA 11.4 + GCC 10 + Python 3.10 + PyTorch 2.4.1)
	docker build \
		-t $(BUILDER_IMAGE) \
		-f builder/Dockerfile \
		--build-arg JOBS=$(JOBS) \
		$(REPO_ROOT)

GIT_SHA := $(shell git -C $(REPO_ROOT) rev-parse HEAD 2>/dev/null || echo unknown)

build-local: ## Build runtime image from local source
	docker build \
		-t $(RUNTIME_LOCAL) \
		-f runtime/Dockerfile.local \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg MAX_JOBS=$(JOBS) \
		--build-arg GIT_SHA=$(GIT_SHA) \
		$(REPO_ROOT)

build-local-no-cache: ## Build runtime image from local source (no layer cache)
	docker build --no-cache \
		-t $(RUNTIME_LOCAL) \
		-f runtime/Dockerfile.local \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg MAX_JOBS=$(JOBS) \
		--build-arg GIT_SHA=$(GIT_SHA) \
		$(REPO_ROOT)

build-all-local: build-builder build-local ## Build builder + runtime (local source)

run: ## Start the vllm37 container and wait until it serves
	./container-up.sh

stop: ## Stop and remove the vllm37 container
	./container-down.sh

logs: ## Show vLLM server logs
	docker compose logs -f

clean: ## Remove builder and runtime images
	-docker rmi $(RUNTIME_LOCAL) $(BUILDER_IMAGE) 2>/dev/null
	@echo "Cleaned up images."

# a git tag of CUTLASS_PIN_REPO
CUTLASS_PIN_TAG ?= v4.0.0
# a git URL
CUTLASS_PIN_REPO ?= https://github.com/NVIDIA/cutlass.git

verify-cutlass-patches: ## Verify cutlass-patches/*.patch apply cleanly to the pinned CUTLASS
	@set -e; \
	WORK=$$(mktemp -d); \
	trap "rm -rf $$WORK" EXIT; \
	echo "fetching CUTLASS $(CUTLASS_PIN_TAG)..."; \
	git clone --depth 1 --branch $(CUTLASS_PIN_TAG) $(CUTLASS_PIN_REPO) $$WORK/cutlass >/dev/null; \
	echo "checking patches against $(CUTLASS_PIN_TAG):"; \
	for p in cutlass-patches/*.patch; do \
	    [ -f "$$p" ] || continue; \
	    if git -C $$WORK/cutlass apply --check "$(REPO_ROOT)/docker/k80/$$p" 2>/dev/null; then \
	        echo "  ✓ $$p"; \
	    else \
	        echo "  ✗ $$p — does not apply cleanly to $(CUTLASS_PIN_TAG)"; \
	        exit 1; \
	    fi; \
	done; \
	echo "all patches apply cleanly to CUTLASS $(CUTLASS_PIN_TAG)"

# a git tag of XFORMERS_PIN_REPO
XFORMERS_PIN_TAG ?= v0.0.23
# a git URL
XFORMERS_PIN_REPO ?= https://github.com/facebookresearch/xformers.git

verify-xformers-patches: ## Verify xformers-patches/*.patch apply cleanly to the pinned XFormers
	@set -e; \
	WORK=$$(mktemp -d); \
	trap "rm -rf $$WORK" EXIT; \
	echo "fetching XFormers $(XFORMERS_PIN_TAG)..."; \
	git clone --depth 1 --branch $(XFORMERS_PIN_TAG) $(XFORMERS_PIN_REPO) $$WORK/xformers >/dev/null; \
	echo "checking patches against $(XFORMERS_PIN_TAG):"; \
	for p in xformers-patches/*.patch; do \
	    [ -f "$$p" ] || continue; \
	    if git -C $$WORK/xformers apply --check "$(REPO_ROOT)/docker/k80/$$p" 2>/dev/null; then \
	        echo "  ✓ $$p"; \
	    else \
	        echo "  ✗ $$p — does not apply cleanly to $(XFORMERS_PIN_TAG)"; \
	        exit 1; \
	    fi; \
	done; \
	echo "all patches apply cleanly to XFormers $(XFORMERS_PIN_TAG)"
