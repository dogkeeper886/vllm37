#!/bin/bash
# Build vllm37 for Tesla K80 (sm_37) on the host filesystem.
#
# Needs on the host: NVIDIA driver 470, CUDA 11.4 at $CUDA_HOME, GCC 10 at
# $CC/$CXX, CMake >= 3.26, git, and the Python build headers (Rocky 8:
# openssl-devel libffi-devel bzip2-devel xz-devel zlib-devel readline-devel
# sqlite-devel tk-devel gdbm-devel ncurses-devel lz4-devel).
#
# Steps run only when their output is missing, so the script can be re-run.
#   1. Python 3.10.16 from source  -> $K80_PREFIX/python3.10
#   2. virtualenv                  -> $K80_VENV
#   3. cuDNN 8.7 for CUDA 11       -> $K80_PREFIX/cudnn
#   4. PyTorch v2.0.1 sm_37 wheel  -> $K80_PREFIX/wheels (about 1-2 h)
#   5. patched xformers v0.0.23    (docker/k80/xformers-build/build.sh)
#   6. vLLM dependencies + vllm37 as an editable install
#
# Env: JOBS (default 7), K80_PREFIX, K80_VENV, CUDA_HOME, CC, CXX.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"
JOBS=${JOBS:-7}
PY_VERSION=3.10.16
TORCH_TAG=v2.0.1
CUDNN_WHEEL="nvidia-cudnn-cu11==8.7.0.84"
SRC=$K80_PREFIX/src
mkdir -p "$SRC" "$K80_PREFIX/wheels"

# 1. Python
if [ ! -x "$K80_PREFIX/python3.10/bin/python3.10" ]; then
  echo "=== 1. Python $PY_VERSION ==="
  cd "$SRC"
  [ -f "Python-$PY_VERSION.tgz" ] || wget -q "https://www.python.org/ftp/python/$PY_VERSION/Python-$PY_VERSION.tgz"
  tar xf "Python-$PY_VERSION.tgz"
  cd "Python-$PY_VERSION"
  ./configure --prefix="$K80_PREFIX/python3.10" --enable-shared --enable-optimizations --with-lto \
    LDFLAGS="-Wl,-rpath=$K80_PREFIX/python3.10/lib"
  make -j"$JOBS" && make install
fi

# 2. venv
if [ ! -x "$K80_VENV/bin/python" ]; then
  echo "=== 2. venv $K80_VENV ==="
  "$K80_PREFIX/python3.10/bin/python3.10" -m venv "$K80_VENV"
fi
source "$K80_VENV/bin/activate"
pip install -q -U pip
pip install -q -c "$HERE/constraints.txt" numpy packaging setuptools wheel ninja pyyaml \
  typing_extensions cffi future six requests dataclasses filelock jinja2 networkx sympy

# 3. cuDNN 8.7 for CUDA 11 (support matrix lists SM 3.5 and later)
if [ ! -f "$K80_PREFIX/cudnn/lib/libcudnn.so.8" ]; then
  echo "=== 3. cuDNN ==="
  DL=$(mktemp -d)
  pip download -q --no-deps "$CUDNN_WHEEL" -d "$DL"
  python -I -m zipfile -e "$DL"/nvidia_cudnn_cu11-*.whl "$DL/x"
  mkdir -p "$K80_PREFIX/cudnn"
  cp -r "$DL/x/nvidia/cudnn/include" "$DL/x/nvidia/cudnn/lib" "$K80_PREFIX/cudnn/"
fi
ln -sf libcudnn.so.8 "$K80_PREFIX/cudnn/lib/libcudnn.so"

# 4. PyTorch
TORCH_WHEEL=$(ls "$K80_PREFIX"/wheels/torch-2.0.1-*.whl 2>/dev/null | head -1 || true)
if [ -z "$TORCH_WHEEL" ]; then
  echo "=== 4. PyTorch $TORCH_TAG ==="
  [ -d "$SRC/pytorch" ] || git clone --depth 1 --branch "$TORCH_TAG" --recursive --shallow-submodules \
    https://github.com/pytorch/pytorch.git "$SRC/pytorch"
  cd "$SRC/pytorch"
  pip install -q -c "$HERE/constraints.txt" -r requirements.txt
  # The tag's version.txt says 2.0.0a0; label the wheel with the real release.
  PYTORCH_BUILD_VERSION=2.0.1 PYTORCH_BUILD_NUMBER=1 \
  CMAKE_PREFIX_PATH=/usr/local CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  USE_CUDA=1 USE_CUDNN=1 USE_NCCL=1 USE_DISTRIBUTED=1 USE_MKLDNN=0 BUILD_TEST=0 \
  MAX_JOBS="$JOBS" python setup.py bdist_wheel
  cp dist/torch-2.0.1*.whl "$K80_PREFIX/wheels/"
  TORCH_WHEEL=$(ls "$K80_PREFIX"/wheels/torch-2.0.1-*.whl | head -1)
fi
python -c "import torch" 2>/dev/null || pip install -q --no-deps "$TORCH_WHEEL"

# 5. xformers
if ! python -c "import xformers.ops" 2>/dev/null; then
  echo "=== 5. xformers ==="
  WORK_DIR="$SRC/xformers-build" MAX_JOBS="$JOBS" bash "$REPO_ROOT/docker/k80/xformers-build/build.sh"
fi

# 6. vLLM
echo "=== 6. vllm37 (editable) ==="
cd "$REPO_ROOT"
pip install -q "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8.0"
pip install -q -c "$HERE/constraints.txt" -r requirements/common.txt "ray>=2.9.0"
VLLM_BUILD_LEGACY_CUDA=1 CMAKE_POLICY_VERSION_MINIMUM=3.5 MAX_JOBS="$JOBS" \
  pip install -e . --no-build-isolation --no-deps
python -c "import torch, vllm; print('torch', torch.__version__, torch.cuda.get_arch_list(), 'vllm', vllm.__version__)"
