#!/bin/bash
# Build vllm37 for Tesla K80 (sm_37). The host build and the docker/k80 images
# both run this script, so they build the same stack.
#
# Usage: setup.sh [STAGE...]      (no stage = all four, in order)
#   toolchain  Python 3.10.16, virtualenv, cuDNN 8.7 for CUDA 11,
#              PyTorch v$TORCH_VERSION sm_37 wheel (about 1-2 h)
#   deps       vLLM's Python packages: docker/k80/requirements.txt
#   xformers   patched xformers v0.0.23 (docker/k80/xformers-build/build.sh)
#   vllm       vllm37 from this checkout
# A step whose output already exists is skipped.
#
# Needs on the host: NVIDIA driver 470, CUDA 11.4 at $CUDA_HOME, GCC 10 at
# $CC/$CXX, CMake >= 3.26, git, wget, and the Python build headers (Rocky 8:
# openssl-devel libffi-devel bzip2-devel xz-devel zlib-devel readline-devel
# sqlite-devel tk-devel gdbm-devel ncurses-devel lz4-devel).
#
# Env:
#   JOBS          parallel compile jobs (default 7)
#   K80_PREFIX    Python, cuDNN, sources, wheels (default ~/opt/k80)
#   K80_VENV      virtualenv (default ~/.venvs/vllm37)
#   K80_EDITABLE  1 = editable vLLM install for development, 0 = regular (default 1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"
JOBS=${JOBS:-7}
K80_EDITABLE=${K80_EDITABLE:-1}
CONSTRAINTS="$REPO_ROOT/docker/k80/constraints.txt"
PY_VERSION=3.10.16
TORCH_VERSION=2.4.1
TORCH_TAG=v$TORCH_VERSION
CUDNN_WHEEL="nvidia-cudnn-cu11==8.7.0.84"
XFORMERS_VERSION=0.0.23+k80
SRC=$K80_PREFIX/src

stage_toolchain() {
  mkdir -p "$SRC" "$K80_PREFIX/wheels"

  if [ ! -x "$K80_PREFIX/python3.10/bin/python3.10" ]; then
    echo "=== toolchain: Python $PY_VERSION ==="
    cd "$SRC"
    [ -f "Python-$PY_VERSION.tgz" ] || wget -q "https://www.python.org/ftp/python/$PY_VERSION/Python-$PY_VERSION.tgz"
    tar xf "Python-$PY_VERSION.tgz"
    cd "Python-$PY_VERSION"
    ./configure --prefix="$K80_PREFIX/python3.10" --enable-shared --enable-optimizations --with-lto \
      LDFLAGS="-Wl,-rpath=$K80_PREFIX/python3.10/lib"
    make -j"$JOBS" && make install
  fi

  if [ ! -x "$K80_VENV/bin/python" ]; then
    echo "=== toolchain: venv $K80_VENV ==="
    "$K80_PREFIX/python3.10/bin/python3.10" -m venv "$K80_VENV"
  fi
  source "$K80_VENV/bin/activate"
  pip install -q -U pip
  pip install -q -c "$CONSTRAINTS" numpy packaging setuptools wheel ninja pyyaml \
    typing_extensions cffi future six requests dataclasses filelock jinja2 networkx sympy

  # cuDNN 8.7 for CUDA 11: its support matrix lists SM 3.5 and later.
  if [ ! -f "$K80_PREFIX/cudnn/lib/libcudnn.so.8" ]; then
    echo "=== toolchain: cuDNN ==="
    local dl
    dl=$(mktemp -d)
    pip download -q --no-deps "$CUDNN_WHEEL" -d "$dl"
    python -I -m zipfile -e "$dl"/nvidia_cudnn_cu11-*.whl "$dl/x"
    mkdir -p "$K80_PREFIX/cudnn"
    cp -r "$dl/x/nvidia/cudnn/include" "$dl/x/nvidia/cudnn/lib" "$K80_PREFIX/cudnn/"
    rm -r "$dl"
  fi
  ln -sf libcudnn.so.8 "$K80_PREFIX/cudnn/lib/libcudnn.so"

  local wheel
  wheel=$(ls "$K80_PREFIX"/wheels/torch-"$TORCH_VERSION"-*.whl 2>/dev/null | head -1 || true)
  if [ -z "$wheel" ]; then
    echo "=== toolchain: PyTorch $TORCH_TAG ==="
    [ -d "$SRC/pytorch-$TORCH_VERSION" ] || git clone --depth 1 --branch "$TORCH_TAG" --recursive --shallow-submodules \
      https://github.com/pytorch/pytorch.git "$SRC/pytorch-$TORCH_VERSION"
    cd "$SRC/pytorch-$TORCH_VERSION"
    pip install -q -c "$CONSTRAINTS" -r requirements.txt
    # A tag's version.txt carries a dev label (e.g. 2.0.0a0); label the wheel
    # with the real release. Flash and memory-efficient attention kernels need
    # sm_50+ and are not built for sm_37.
    PYTORCH_BUILD_VERSION="$TORCH_VERSION" PYTORCH_BUILD_NUMBER=1 \
    CMAKE_PREFIX_PATH=/usr/local CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    USE_CUDA=1 USE_CUDNN=1 USE_NCCL=1 USE_DISTRIBUTED=1 USE_MKLDNN=0 BUILD_TEST=0 \
    USE_FLASH_ATTENTION=0 USE_MEM_EFFICIENT_ATTENTION=0 \
    MAX_JOBS="$JOBS" python setup.py bdist_wheel
    cp dist/torch-"$TORCH_VERSION"*.whl "$K80_PREFIX/wheels/"
    wheel=$(ls "$K80_PREFIX"/wheels/torch-"$TORCH_VERSION"-*.whl | head -1)
  fi
  # Leave the source tree: from inside it, `import torch` finds the unbuilt package.
  cd "$K80_PREFIX"
  python -c "import torch" 2>/dev/null || pip install -q --no-deps "$wheel"
  python -c "import torch; print('torch', torch.__version__, 'cuDNN', torch.backends.cudnn.version())"
}

stage_deps() {
  echo "=== deps ==="
  source "$K80_VENV/bin/activate"
  pip install -q -c "$CONSTRAINTS" -r "$REPO_ROOT/docker/k80/requirements.txt"
  # xgrammar pulls in Triton, which cannot compile for sm_37. vLLM falls back
  # to its Triton placeholder when it is absent.
  pip uninstall -y -q triton 2>/dev/null || true
}

stage_xformers() {
  source "$K80_VENV/bin/activate"
  if python -c "import xformers.ops" 2>/dev/null; then
    return
  fi
  echo "=== xformers ==="
  BUILD_VERSION="$XFORMERS_VERSION" WORK_DIR="$SRC/xformers-build" MAX_JOBS="$JOBS" \
    bash "$REPO_ROOT/docker/k80/xformers-build/build.sh"
}

stage_vllm() {
  echo "=== vllm37 (editable=$K80_EDITABLE) ==="
  source "$K80_VENV/bin/activate"
  cd "$REPO_ROOT"
  pip install -q -c "$CONSTRAINTS" "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8.0"
  local editable=()
  [ "$K80_EDITABLE" = 1 ] && editable=(-e)
  VLLM_BUILD_LEGACY_CUDA=1 CMAKE_POLICY_VERSION_MINIMUM=3.5 MAX_JOBS="$JOBS" \
    pip install "${editable[@]}" . --no-build-isolation --no-deps
  python -c "import torch, vllm; print('torch', torch.__version__, torch.cuda.get_arch_list(), 'vllm', vllm.__version__)"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(toolchain deps xformers vllm)
for s in "${stages[@]}"; do
  case "$s" in
    toolchain|deps|xformers|vllm) "stage_$s" ;;
    *) echo "unknown stage: $s (expected toolchain, deps, xformers, vllm)" >&2; exit 2 ;;
  esac
done
