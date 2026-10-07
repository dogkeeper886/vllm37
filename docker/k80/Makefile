# vLLM K80 Build System
# Follows the dogkeeper886/ollama37 Makefile pattern.
#
# Usage:
#   make build-builder    # Build the builder image (~120 min first time)
#   make build-local      # Build the runtime image from this checkout
#   make build-all-local  # Build builder + runtime
#   make run              # Run with docker compose
#   make clean            # Remove images

-include .env

BUILDER_IMAGE  ?= vllm37-builder
RUNTIME_LOCAL  ?= vllm37-local
JOBS           ?= 4
# Host dir bind-mounted into the container at /var/log/vllm for vllm + NCCL
# logs. Must exist and be writable before `docker compose up` — otherwise
# Docker auto-creates it as root-owned and NCCL_DEBUG_FILE silently fails.
VLLM_LOG_DIR   ?= /tmp/vllm-logs
REPO_ROOT      := $(shell git rev-parse --show-toplevel 2>/dev/null || echo ../..)

.PHONY: help build-builder build-local build-local-no-cache build-all-local log-dir run stop logs clean verify-cutlass-patches verify-xformers-patches

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build-builder: ## Build the builder image (CUDA 11.4 + GCC 10 + Python 3.10 + PyTorch 2.0.1)
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

log-dir: ## Ensure VLLM_LOG_DIR exists (bind-mount target for vllm + NCCL logs)
	@mkdir -p $(VLLM_LOG_DIR)
	@chmod 777 $(VLLM_LOG_DIR)

run: log-dir ## Start vLLM server via docker compose
	VLLM_LOG_DIR=$(VLLM_LOG_DIR) docker compose up -d

stop: ## Stop vLLM server
	docker compose down

logs: ## Show vLLM server logs
	docker compose logs -f

clean: ## Remove builder and runtime images
	-docker rmi $(RUNTIME_LOCAL) $(BUILDER_IMAGE) 2>/dev/null
	@echo "Cleaned up images."

# CUTLASS revision pinned by CMakeLists.txt:302 — keep these in sync if upstream
# vLLM bumps the FetchContent revision.
CUTLASS_PIN_TAG ?= v4.0.0
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

# XFormers pin — v0.0.23 (Dec 2023). Two simultaneous constraints force this:
#   1. v0.0.29+ removed the cutlass mem-eff backend (commit 3d947a6)
#   2. v0.0.24+ requires torch>=2.1; v0.0.28+ requires torch>=2.4. Our K80
#      fork uses PyTorch 2.0.1 (docker/k80/constraints.txt), so we cap at
#      v0.0.23 (torch>=1.12).
# v0.0.23 is the latest tag that satisfies BOTH constraints. See
# xformers-patches/README.md for the full rationale.
XFORMERS_PIN_TAG ?= v0.0.23
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
