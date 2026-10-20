# Env:
#   K80_PREFIX  a directory (default ~/opt/k80)
#   K80_VENV    a virtualenv directory (default ~/.venvs/vllm37)
#   CUDA_HOME   a CUDA 11.4 directory (default /usr/local/cuda-11.4)
#   CC          a GCC 10 gcc (default /usr/local/bin/gcc)
#   CXX         a GCC 10 g++ (default /usr/local/bin/g++)
export K80_PREFIX=${K80_PREFIX:-$HOME/opt/k80}
export K80_VENV=${K80_VENV:-$HOME/.venvs/vllm37}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.4}
export CC=${CC:-/usr/local/bin/gcc}
export CXX=${CXX:-/usr/local/bin/g++}
export TORCH_CUDA_ARCH_LIST="3.7"
export CUDNN_INCLUDE_DIR=$K80_PREFIX/cudnn/include
export CUDNN_LIBRARY=$K80_PREFIX/cudnn/lib
export CUDNN_ROOT_DIR=$K80_PREFIX/cudnn
export PATH=$CUDA_HOME/bin:/usr/local/bin:$PATH
export LD_LIBRARY_PATH=$K80_PREFIX/cudnn/lib:$CUDA_HOME/lib64:/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [ -f "$K80_VENV/bin/activate" ]; then
  source "$K80_VENV/bin/activate"
fi
