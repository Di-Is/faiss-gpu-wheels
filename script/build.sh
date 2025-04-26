#!/usr/bin/env bash

set -eux

# Enable yum cache
YUM_CACHE_DIR='/host/tmp/.cache/yum'
cat <<EOF >/etc/yum.conf
[main]
cachedir=${YUM_CACHE_DIR}
keepcache=1
EOF
[ ! -d "$YUM_CACHE_DIR" ] && mkdir -p "$YUM_CACHE_DIR" && yum makecache

# Install blas
yum install -y openblas-devel openblas-static

FAISS_ENABLE_GPU=${FAISS_ENABLE_GPU:-"OFF"}
FAISS_ENABLE_ROCM=${FAISS_ENABLE_ROCM:-"OFF"}
FAISS_ENABLE_RAFT=${FAISS_ENABLE_RAFT:-"OFF"}
if [ "$FAISS_ENABLE_GPU" == "ON" ]; then
    if [ "$FAISS_ENABLE_ROCM" != "ON" ]; then
        CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 1)
        CUDA_MINOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 2)
        # Install CUDA
        yum -y install yum-utils
        yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
        yum -y install cuda-toolkit-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}-${CUDA_VERSION}
        ln -s cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} /usr/local/cuda
        export PATH="/usr/local/cuda/bin:${PATH}"
    fi
fi

# Build and patch faiss
FAISS_OPT_LEVEL=${FAISS_OPT_LEVEL:-"generic"}
CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-"70-real;80-real"}
NJOB=${NJOB:-"$(nproc)"}
cd faiss
cmake . -B build \
    -DFAISS_OPT_LEVEL=${FAISS_OPT_LEVEL} \
    -DFAISS_ENABLE_GPU=${FAISS_ENABLE_GPU} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -DFAISS_ENABLE_RAFT=${FAISS_ENABLE_RAFT} \
    -DFAISS_ENABLE_ROCM=${FAISS_ENABLE_ROCM} \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j${NJOB}
cmake --install build
cd ..
git apply patch/modify-numpy-find-package.patch
