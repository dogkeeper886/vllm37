#!/usr/bin/env bash
# Env:
#   PATCH_DIR     a directory of CUTLASS .patch files (default
#                 ../cutlass-patches beside this script)
#   WORK_DIR      a directory (default /tmp/cutlass-sm37-build)
#   CUTLASS_TAG   a git tag of CUTLASS_REPO that the patches apply to
#                 (default v4.0.0)
#   CUTLASS_REPO  a git URL (default https://github.com/NVIDIA/cutlass.git)
#   NVCC          an nvcc command or path (default nvcc)
#   TARGET_ARCH   an nvcc -arch value such as sm_37 (default sm_37)

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
