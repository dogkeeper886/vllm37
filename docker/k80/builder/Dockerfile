# vLLM K80 Builder Image
# Rocky Linux 8 with CUDA 11.4, GCC 10.5 and CMake 4, plus the toolchain stage
# of tools/k80-host/setup.sh: Python 3.10.16, a virtualenv at /opt/venv,
# cuDNN 8.7 for CUDA 11 and a PyTorch v2.4.1 sm_37 wheel. The host build runs
# the same script, so both get the same stack.
#
# Build (from the repo root; the Makefile does this):
#   docker build -t vllm37-builder -f docker/k80/builder/Dockerfile .
#
# Expected build time: ~120 min (GCC ~60 min, Python ~15 min, PyTorch ~45 min)

FROM rockylinux/rockylinux:8

ARG GCC_VERSION=10.5.0
ARG CMAKE_VERSION=4.0.1
ARG JOBS=4

# ============================================================================
# System packages
# ============================================================================
RUN dnf -y install \
        dnf-plugins-core \
        epel-release && \
    dnf config-manager --set-enabled powertools && \
    dnf -y install \
        bzip2 bzip2-devel \
        diffutils \
        file \
        findutils \
        gdbm-devel \
        git \
        libffi-devel \
        lz4-devel \
        make \
        ncurses-devel \
        openssl-devel \
        patch \
        pciutils \
        readline-devel \
        sqlite-devel \
        tar \
        tk-devel \
        unzip \
        wget \
        which \
        xz xz-devel \
        zlib-devel \
    && dnf clean all

# ============================================================================
# CUDA 11.4 Toolkit (cuDNN comes from the toolchain stage below: the repo's
# libcudnn8 resolves to a CUDA 12 build that driver 470 cannot run)
# ============================================================================
RUN dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf -y install \
        cuda-toolkit-11-4 \
    && dnf clean all

ENV CUDA_HOME=/usr/local/cuda-11.4
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ============================================================================
# GCC 10 from source (Rocky 8 ships GCC 8, too old for PyTorch 2.x)
# ============================================================================
RUN cd /tmp && \
    wget -q https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz && \
    tar xf gcc-${GCC_VERSION}.tar.xz && \
    cd gcc-${GCC_VERSION} && \
    ./contrib/download_prerequisites && \
    mkdir build && cd build && \
    ../configure \
        --prefix=/usr/local \
        --enable-languages=c,c++ \
        --disable-multilib \
        --disable-bootstrap && \
    make -j${JOBS} && \
    make install && \
    cd /tmp && rm -rf gcc-${GCC_VERSION}*

ENV CC=/usr/local/bin/gcc
ENV CXX=/usr/local/bin/g++
ENV LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH}

# Update linker cache for new GCC
RUN echo "/usr/local/lib64" > /etc/ld.so.conf.d/local-lib64.conf && ldconfig

# ============================================================================
# CMake 4 from source
# ============================================================================
RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar xf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap --prefix=/usr/local --parallel=${JOBS} && \
    make -j${JOBS} && \
    make install && \
    cd /tmp && rm -rf cmake-${CMAKE_VERSION}*

# ============================================================================
# Toolchain stage: Python 3.10.16, venv, cuDNN 8.7, PyTorch v2.4.1 (sm_37)
# ============================================================================
ENV K80_PREFIX=/opt/k80 \
    K80_VENV=/opt/venv
COPY tools/k80-host /usr/local/src/vllm37/tools/k80-host
COPY docker/k80/constraints.txt /usr/local/src/vllm37/docker/k80/constraints.txt
RUN JOBS=${JOBS} /usr/local/src/vllm37/tools/k80-host/setup.sh toolchain && \
    rm -rf ${K80_PREFIX}/src

ENV PATH=${K80_VENV}/bin:${PATH} \
    LD_LIBRARY_PATH=${K80_PREFIX}/cudnn/lib:${LD_LIBRARY_PATH} \
    CUDNN_INCLUDE_DIR=${K80_PREFIX}/cudnn/include \
    CUDNN_LIBRARY=${K80_PREFIX}/cudnn/lib

WORKDIR /usr/local/src
