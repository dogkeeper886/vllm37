FROM rockylinux/rockylinux:8

# a GCC release from 9.3.0 to 11.5.0
ARG GCC_VERSION=10.5.0
# a CMake release from 3.26.0
ARG CMAKE_VERSION=4.0.1
# a whole number from 1
ARG JOBS=4

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

RUN dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf -y install \
        cuda-toolkit-11-4 \
    && dnf clean all

ENV CUDA_HOME=/usr/local/cuda-11.4
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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

RUN echo "/usr/local/lib64" > /etc/ld.so.conf.d/local-lib64.conf && ldconfig

RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar xf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap --prefix=/usr/local --parallel=${JOBS} && \
    make -j${JOBS} && \
    make install && \
    cd /tmp && rm -rf cmake-${CMAKE_VERSION}*

# K80_PREFIX: a directory
# K80_VENV: a directory
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
