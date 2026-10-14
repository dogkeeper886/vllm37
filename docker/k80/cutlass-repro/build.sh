#!/usr/bin/env bash
#
# Build the sm_37 CUTLASS SGEMM reproducers: a manual check of
# cutlass-patches/sm37-trait.patch, the CUTLASS patch the xformers build uses.
#   sgemm_sm37            one SGEMM through arch::Sm37, checked against a
#                         deterministic input
#   sgemm_sm37_vs_cublas  the same SGEMM through CUTLASS Sm37 and cuBLAS at
#                         several sizes, compared element-wise
#
# Designed to run inside the K80 builder image (vllm37-builder:latest), which
# already has CUDA 11.4 + GCC 10 + CMake 4 + Python 3.10. The script:
#
#   1. Clones CUTLASS at the FetchContent-pinned tag (v4.0.0 per
#      CMakeLists.txt:302) into a scratch directory.
#   2. Applies sm37-trait.patch.
#   3. Compiles both .cu files with nvcc, targeting sm_37.
#   4. Prints the resulting binary paths + sizes for the caller.
#
# The runtime execution is intentionally NOT done here. No workflow runs the
# binaries now; run them by hand on the host.
#
# Idempotent: re-running re-applies the patch via `git apply -3` so an
# already-patched tree doesn't trip up subsequent builds.

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# All env vars below respect caller-supplied overrides (`:-` defaulting).
# This matters when a caller mounts the patches dir elsewhere, such as
# /patches inside a container, NOT at $REPRO_DIR/../cutlass-patches.
PATCH_DIR="${PATCH_DIR:-${REPRO_DIR}/../cutlass-patches}"
WORK_DIR="${WORK_DIR:-/tmp/cutlass-sm37-build}"
CUTLASS_TAG="${CUTLASS_TAG:-v4.0.0}"
CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
NVCC="${NVCC:-nvcc}"
TARGET_ARCH="${TARGET_ARCH:-sm_37}"

echo "=== Build env ==="
echo "REPRO_DIR     = ${REPRO_DIR}"
echo "PATCH_DIR     = ${PATCH_DIR}"
echo "WORK_DIR      = ${WORK_DIR}"
echo "CUTLASS_TAG   = ${CUTLASS_TAG}"
echo "TARGET_ARCH   = ${TARGET_ARCH}"
echo "NVCC          = ${NVCC}"
"${NVCC}" --version | head -4 || true
echo ""

echo "=== Step 1: prepare CUTLASS source at ${CUTLASS_TAG} ==="
if [ -d "${WORK_DIR}/cutlass" ] && [ -d "${WORK_DIR}/cutlass/.git" ]; then
    echo "found existing CUTLASS clone at ${WORK_DIR}/cutlass — reusing"
    git -C "${WORK_DIR}/cutlass" fetch --depth 1 origin "${CUTLASS_TAG}"
    git -C "${WORK_DIR}/cutlass" reset --hard "FETCH_HEAD"
else
    rm -rf "${WORK_DIR}/cutlass"
    mkdir -p "${WORK_DIR}"
    git clone --depth 1 --branch "${CUTLASS_TAG}" "${CUTLASS_REPO}" "${WORK_DIR}/cutlass"
fi
echo ""

echo "=== Step 2: apply sm_37 patches ==="
# Defensive check: PATCH_DIR must exist and contain at least one .patch.
# Previously this loop silently iterated over an empty glob if PATCH_DIR
# was wrong, leaving an unpatched tree to fail later with a confusing
# "Sm37 struct not present" message. Better to fail loudly here.
if [ ! -d "${PATCH_DIR}" ]; then
    echo "ERROR: PATCH_DIR does not exist: ${PATCH_DIR}"
    exit 1
fi
shopt -s nullglob
patch_files=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob
if [ ${#patch_files[@]} -eq 0 ]; then
    echo "ERROR: no .patch files found in ${PATCH_DIR}"
    exit 1
fi
echo "found ${#patch_files[@]} patch(es) in ${PATCH_DIR}:"
for p in "${patch_files[@]}"; do echo "  - $(basename "$p")"; done

# Use `git apply -3` for 3-way merge — handles already-applied AND minor
# context drift if upstream CUTLASS adds nearby content.
for patch_file in "${patch_files[@]}"; do
    patch_name="$(basename "${patch_file}")"
    echo "applying ${patch_name}..."
    git -C "${WORK_DIR}/cutlass" apply -3 "${patch_file}" \
        || git -C "${WORK_DIR}/cutlass" apply --check --reverse "${patch_file}" \
        || { echo "patch ${patch_name} failed and is not already applied"; exit 1; }
done
echo "verifying Sm37 struct present:"
grep -A1 "^struct Sm37" "${WORK_DIR}/cutlass/include/cutlass/arch/arch.h" \
    || { echo "ERROR: Sm37 struct not present after patching"; exit 1; }
echo ""

echo "=== Step 3a: compile sgemm_sm37.cu for ${TARGET_ARCH} ==="
mkdir -p "${WORK_DIR}/build"
"${NVCC}" \
    -std=c++17 \
    -O3 \
    -arch="${TARGET_ARCH}" \
    -I "${WORK_DIR}/cutlass/include" \
    -I "${WORK_DIR}/cutlass/tools/util/include" \
    -Xcompiler -Wall \
    "${REPRO_DIR}/sgemm_sm37.cu" \
    -o "${WORK_DIR}/build/sgemm_sm37"
ls -la "${WORK_DIR}/build/sgemm_sm37"
echo ""

echo "=== Step 3b: compile sgemm_sm37_vs_cublas.cu for ${TARGET_ARCH} ==="
# Same SGEMM via CUTLASS Sm37 and cuBLAS, compared element-wise.
# Needs -lcublas at link time.
"${NVCC}" \
    -std=c++17 \
    -O3 \
    -arch="${TARGET_ARCH}" \
    -I "${WORK_DIR}/cutlass/include" \
    -I "${WORK_DIR}/cutlass/tools/util/include" \
    -Xcompiler -Wall \
    "${REPRO_DIR}/sgemm_sm37_vs_cublas.cu" \
    -lcublas \
    -o "${WORK_DIR}/build/sgemm_sm37_vs_cublas"
ls -la "${WORK_DIR}/build/sgemm_sm37_vs_cublas"
echo ""

echo "=== Build complete ==="
echo "binaries:"
echo "  - ${WORK_DIR}/build/sgemm_sm37"
echo "  - ${WORK_DIR}/build/sgemm_sm37_vs_cublas"
