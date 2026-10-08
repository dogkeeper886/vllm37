#!/usr/bin/env bash
#
# Build XFormers v0.0.23 from source against patched CUTLASS, targeting sm_37
# (Tesla K80). Story #33 of epic #12.
#
# Designed to run inside the K80 builder image (vllm37-builder:latest), which
# has CUDA 11.4 + GCC 10 + PyTorch 2.4.1 + CMake 4 + Python 3.10. The script:
#
#   1. Clones XFormers v0.0.23 with submodules (recursive — pulls bundled
#      CUTLASS at e0aaa3c3 / Sep 2023 / pre-v3.5).
#   2. Applies XFormers patches (xformers-patches/*.patch).
#   3. Applies CUTLASS patches to the bundled submodule
#      (cutlass-patches/*.patch — same patches Phase 1 verified).
#   4. Builds XFormers with TORCH_CUDA_ARCH_LIST="3.7" so NVCC emits sm_37
#      kernels.
#   5. Verifies the install (xformers._C imports; FwOp.CUDA_MINIMUM_COMPUTE_
#      CAPABILITY reflects our patch).
#
# Idempotent: re-applies patches via `git apply -3` so a previously-patched
# tree doesn't trip up subsequent runs.

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

# -----------------------------------------------------------------------------
# Step 1: prepare XFormers source
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Step 2: apply XFormers patches (sm_37)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Step 2c: regenerate autogen kernels (with sm_37)
# -----------------------------------------------------------------------------
# CRITICAL: the autogen .cu files are committed to the XFormers repo as
# pre-generated source. setup.py does NOT invoke generate_kernels.py at build
# time — running it is a manual developer step that produces the committed
# files. So our sm37-generate-kernels.patch updates the SCRIPT but not the
# generated output. We need to run the script ourselves to regenerate the
# autogen files with sm_37 instantiations included.
#
# Symptom of skipping this: build succeeds, .so loads, dispatcher accepts the
# device, but kernel launch fails with "cutlassF: no kernel found to launch!"
# because the compiled .so contains _sm50/_sm70/_sm75/_sm80 symbols but no
# _sm37 (PR #64 review run 24959810424 caught this).
echo "=== Step 2c: regenerate autogen kernels (with sm_37) ==="
cd "${WORK_DIR}/xformers"
python xformers/csrc/attention/cuda/fmha/generate_kernels.py
# `|| sm37_count=0` defends against `set -o pipefail` — if grep finds zero
# matches it exits 1, which would otherwise propagate through the pipeline,
# trip set -e on the assignment, and abort the script before the if-check
# runs. We want the explicit "expected ≥50" diagnostic to fire instead.
sm37_count=$(grep -rE "Sm37|sm37" xformers/csrc/attention/cuda/fmha/autogen/ 2>/dev/null | wc -l) || sm37_count=0
echo "regenerated autogen files contain ${sm37_count} sm_37 references"
if [ "${sm37_count}" -lt 50 ]; then
    echo "ERROR: expected ≥50 sm_37 references after autogen; got ${sm37_count}"
    echo "       (the patched generate_kernels.py may not have produced sm_37 variants)"
    exit 1
fi
echo ""

# -----------------------------------------------------------------------------
# Step 2b: apply CUTLASS patches to bundled submodule
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Step 3: build XFormers
# -----------------------------------------------------------------------------
echo "=== Step 3: build XFormers (TORCH_CUDA_ARCH_LIST=${TARGET_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}) ==="
echo "this can take 30+ minutes on K80 hardware"
cd "${WORK_DIR}/xformers"

export TORCH_CUDA_ARCH_LIST="${TARGET_ARCH_LIST}"
export MAX_JOBS
export FORCE_CUDA=1
# XFormers reads XFORMERS_DISABLE_TRITON to skip Triton ops (which we don't
# need on K80 — Triton requires sm_70+).
export XFORMERS_DISABLE_TRITON=1

# --no-build-isolation reuses the system Python's installed build deps
# (setuptools, wheel, ninja, pybind11, torch). The K80 builder image already
# has these for vLLM compilation.
#
# No `| tail` here — the workflow captures full stdout/stderr to /out/build.log
# for the artifact. The GitHub-log view is abbreviated by the workflow's own
# `tail -200 /out/build.log` step. Truncating here would lose the full nvcc
# output that we need to diagnose first-run failures on a novel
# (PyTorch 2.4.1 + XFormers v0.0.23 + sm_37) combination.
pip install -v --no-build-isolation . 2>&1

echo ""

# -----------------------------------------------------------------------------
# Step 4: verify
# -----------------------------------------------------------------------------
echo "=== Step 4: verify installation ==="
# IMPORTANT: cd OUT of the xformers source directory before importing.
# Python's sys.path puts the CWD first; if we stay in /tmp/.../xformers, the
# source package (no _C.so) shadows the installed one. cd to /tmp to use
# normal site-packages resolution.
cd /tmp
python <<'PYEOF'
import sys
import xformers
print(f"xformers version: {xformers.__version__}")
print(f"xformers location: {xformers.__file__}")

import xformers._C as c
print(f"xformers._C imported OK from: {c.__file__}")

# Confirm our cc-gate patch landed
from xformers.ops.fmha.common import AttentionOpBase
cc = AttentionOpBase.CUDA_MINIMUM_COMPUTE_CAPABILITY
print(f"AttentionOpBase.CUDA_MINIMUM_COMPUTE_CAPABILITY = {cc}")
if cc != (3, 7):
    print(f"WARN: expected (3, 7) base default; got {cc} — cc-gate patch may not have applied")
    sys.exit(1)

# cutlassF op should still exist and inherit
from xformers.ops.fmha.cutlass import FwOp
fwop_cc = FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY
print(f"cutlass.FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY = {fwop_cc}")

print("RESULT: PASS — xformers built and imports cleanly")
PYEOF

echo ""
echo "=== Build complete ==="
echo "xformers installed inside the container at: $(python -c 'import xformers; print(xformers.__file__)')"
