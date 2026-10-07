# Host build environment for vllm37 on Tesla K80. Usage: source tools/k80-host/env.sh
# K80_PREFIX holds Python, cuDNN, sources and wheels; K80_VENV is the virtualenv.
export K80_PREFIX=${K80_PREFIX:-$HOME/opt/k80}
export K80_VENV=${K80_VENV:-$HOME/.venvs/vllm37}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.4}
export CC=${CC:-/usr/local/bin/gcc}
export CXX=${CXX:-/usr/local/bin/g++}
export TORCH_CUDA_ARCH_LIST="3.7"
# PyTorch's FindCUDNN searches CUDNN_LIBRARY as a directory, not a file.
export CUDNN_INCLUDE_DIR=$K80_PREFIX/cudnn/include
export CUDNN_LIBRARY=$K80_PREFIX/cudnn/lib
export CUDNN_ROOT_DIR=$K80_PREFIX/cudnn
export PATH=$CUDA_HOME/bin:/usr/local/bin:$PATH
export LD_LIBRARY_PATH=$K80_PREFIX/cudnn/lib:$CUDA_HOME/lib64:/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [ -f "$K80_VENV/bin/activate" ]; then
  source "$K80_VENV/bin/activate"
fi
