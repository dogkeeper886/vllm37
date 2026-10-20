#!/usr/bin/env bash
# Env:
#   PATCH_DIR          a directory of xformers .patch files (default
#                      ../xformers-patches beside this script)
#   CUTLASS_PATCH_DIR  a directory of CUTLASS .patch files (default
#                      ../cutlass-patches beside this script)
#   WORK_DIR           a directory (default /tmp/xformers-sm37-build)
#   XFORMERS_TAG       a git tag of XFORMERS_REPO that the patches apply to
#                      (default v0.0.23)
#   XFORMERS_REPO      a git URL
#                      (default https://github.com/facebookresearch/xformers.git)
#   TARGET_ARCH_LIST   a CUDA arch list such as 3.7 (default 3.7)
#   MAX_JOBS           a whole number from 1 (default 4)

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-${REPRO_DIR}/../xformers-patches}"
CUTLASS_PATCH_DIR="${CUTLASS_PATCH_DIR:-${REPRO_DIR}/../cutlass-patches}"
WORK_DIR="${WORK_DIR:-/tmp/xformers-sm37-build}"
XFORMERS_TAG="${XFORMERS_TAG:-v0.0.23}"
XFORMERS_REPO="${XFORMERS_REPO:-https://github.com/facebookresearch/xformers.git}"
TARGET_ARCH_LIST="${TARGET_ARCH_LIST:-3.7}"
MAX_JOBS="${MAX_JOBS:-4}"

echo "=== Build env ==="
echo "REPRO_DIR         = ${REPRO_DIR}"
echo "PATCH_DIR         = ${PATCH_DIR}"
echo "CUTLASS_PATCH_DIR = ${CUTLASS_PATCH_DIR}"
echo "WORK_DIR          = ${WORK_DIR}"
echo "XFORMERS_TAG      = ${XFORMERS_TAG}"
echo "TARGET_ARCH_LIST  = ${TARGET_ARCH_LIST}"
echo "MAX_JOBS          = ${MAX_JOBS}"
which nvcc && nvcc --version | head -4 || echo "(nvcc not on PATH)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 || true
echo ""

echo "=== Step 1: clone XFormers ${XFORMERS_TAG} (with submodules) ==="
if [ ! -d "${PATCH_DIR}" ]; then
    echo "ERROR: PATCH_DIR does not exist: ${PATCH_DIR}"
    exit 1
fi
if [ ! -d "${CUTLASS_PATCH_DIR}" ]; then
    echo "ERROR: CUTLASS_PATCH_DIR does not exist: ${CUTLASS_PATCH_DIR}"
    exit 1
fi

if [ -d "${WORK_DIR}/xformers/.git" ]; then
    echo "found existing XFormers clone at ${WORK_DIR}/xformers — refreshing"
    git -C "${WORK_DIR}/xformers" fetch --depth 1 origin "+refs/tags/${XFORMERS_TAG}:refs/tags/${XFORMERS_TAG}"
    git -C "${WORK_DIR}/xformers" reset --hard "${XFORMERS_TAG}"
    git -C "${WORK_DIR}/xformers" submodule update --init --recursive --depth 1 --force
else
    rm -rf "${WORK_DIR}/xformers"
    mkdir -p "${WORK_DIR}"
    git clone --depth 1 --branch "${XFORMERS_TAG}" \
        --recurse-submodules --shallow-submodules \
        "${XFORMERS_REPO}" "${WORK_DIR}/xformers"
fi

echo "XFormers HEAD:"
git -C "${WORK_DIR}/xformers" log -1 --pretty=format:"  %H %ad %s" --date=short
echo ""
echo "Bundled CUTLASS submodule:"
git -C "${WORK_DIR}/xformers/third_party/cutlass" log -1 --pretty=format:"  %H %ad %s" --date=short
echo ""

echo "=== Step 2a: apply XFormers patches ==="
shopt -s nullglob
xfmr_patches=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob
if [ ${#xfmr_patches[@]} -eq 0 ]; then
    echo "ERROR: no .patch files found in ${PATCH_DIR}"
    exit 1
fi
for patch_file in "${xfmr_patches[@]}"; do
    patch_name="$(basename "${patch_file}")"
    echo "applying ${patch_name}..."
    git -C "${WORK_DIR}/xformers" apply -3 "${patch_file}" \
        || git -C "${WORK_DIR}/xformers" apply --check --reverse "${patch_file}" \
        || { echo "ERROR: ${patch_name} failed and is not already applied"; exit 1; }
done

echo "verifying Sm37 in generate_kernels.py:"
grep -n "^SM = " "${WORK_DIR}/xformers/xformers/csrc/attention/cuda/fmha/generate_kernels.py" \
    || { echo "ERROR: SM line not found"; exit 1; }

echo "verifying CC gate in common.py:"
grep -n "CUDA_MINIMUM_COMPUTE_CAPABILITY: Tuple" "${WORK_DIR}/xformers/xformers/ops/fmha/common.py" \
    || { echo "ERROR: CC gate line not found"; exit 1; }
echo ""

echo "=== Step 2c: regenerate autogen kernels (with sm_37) ==="
cd "${WORK_DIR}/xformers"
python xformers/csrc/attention/cuda/fmha/generate_kernels.py
sm37_count=$(grep -rE "Sm37|sm37" xformers/csrc/attention/cuda/fmha/autogen/ 2>/dev/null | wc -l) || sm37_count=0
echo "regenerated autogen files contain ${sm37_count} sm_37 references"
if [ "${sm37_count}" -lt 50 ]; then
    echo "ERROR: expected ≥50 sm_37 references after autogen; got ${sm37_count}"
    echo "       (the patched generate_kernels.py may not have produced sm_37 variants)"
    exit 1
fi
echo ""

echo "=== Step 2b: apply CUTLASS patches to bundled submodule ==="
shopt -s nullglob
cutlass_patches=( "${CUTLASS_PATCH_DIR}"/*.patch )
shopt -u nullglob
if [ ${#cutlass_patches[@]} -eq 0 ]; then
    echo "ERROR: no .patch files found in ${CUTLASS_PATCH_DIR}"
    exit 1
fi
for patch_file in "${cutlass_patches[@]}"; do
    patch_name="$(basename "${patch_file}")"
    echo "applying ${patch_name}..."
    git -C "${WORK_DIR}/xformers/third_party/cutlass" apply -3 "${patch_file}" \
        || git -C "${WORK_DIR}/xformers/third_party/cutlass" apply --check --reverse "${patch_file}" \
        || { echo "ERROR: ${patch_name} failed and is not already applied"; exit 1; }
done

echo "verifying Sm37 struct in bundled CUTLASS arch.h:"
grep -A1 "^struct Sm37" "${WORK_DIR}/xformers/third_party/cutlass/include/cutlass/arch/arch.h" \
    || { echo "ERROR: Sm37 struct not present after CUTLASS patching"; exit 1; }
echo ""

echo "=== Step 3: build XFormers (TORCH_CUDA_ARCH_LIST=${TARGET_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}) ==="
cd "${WORK_DIR}/xformers"

export TORCH_CUDA_ARCH_LIST="${TARGET_ARCH_LIST}"
export MAX_JOBS
export FORCE_CUDA=1

pip install -v --no-build-isolation . 2>&1

echo ""

echo "=== Step 4: verify installation ==="
cd /tmp
python <<'PYEOF'
import sys
import xformers
print(f"xformers version: {xformers.__version__}")
print(f"xformers location: {xformers.__file__}")

import xformers._C as c
print(f"xformers._C imported OK from: {c.__file__}")

from xformers.ops.fmha.common import AttentionOpBase
cc = AttentionOpBase.CUDA_MINIMUM_COMPUTE_CAPABILITY
print(f"AttentionOpBase.CUDA_MINIMUM_COMPUTE_CAPABILITY = {cc}")
if cc != (3, 7):
    print(f"WARN: expected (3, 7) base default; got {cc} — cc-gate patch may not have applied")
    sys.exit(1)

from xformers.ops.fmha.cutlass import FwOp
fwop_cc = FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY
print(f"cutlass.FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY = {fwop_cc}")

print("RESULT: PASS — xformers built and imports cleanly")
PYEOF

echo ""
echo "=== Build complete ==="
echo "xformers installed inside the container at: $(python -c 'import xformers; print(xformers.__file__)')"
