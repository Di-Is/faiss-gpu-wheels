#!/usr/bin/env bash

set -eux

# Enable RPM package-manager cache
RPM_CACHE_DIR='/host/tmp/.cache/rpm'
cat <<EOF >/etc/yum.conf
[main]
cachedir=${RPM_CACHE_DIR}
keepcache=1
EOF

if command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER='dnf'
    REPO_SETUP_PACKAGE='dnf-plugins-core'
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER='yum'
    REPO_SETUP_PACKAGE='yum-utils'
else
    echo "Neither dnf nor yum is available." >&2
    exit 1
fi
PKG_INSTALL=("$PKG_MANAGER" install -y)
REPO_MAJOR_VERSION=$(rpm -E '%{rhel}')

mkdir -p "$RPM_CACHE_DIR"
"$PKG_MANAGER" makecache

# Install blas
"${PKG_INSTALL[@]}" openblas-devel openblas-static

# Install ccache (prebuilt binary for NVCC support, RPM versions are too old)
CCACHE_VERSION="${CCACHE_VERSION:-4.13}"
CCACHE_URL="https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64-musl-static.tar.xz"
curl -fsSL "$CCACHE_URL" | tar -xJ --strip-components=1 -C /usr/local/bin/ "ccache-${CCACHE_VERSION}-linux-x86_64-musl-static/ccache"

# Configure ccache
export CCACHE_DIR='/host/tmp/.cache/ccache'
mkdir -p "$CCACHE_DIR"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

FAISS_ENABLE_GPU=${FAISS_ENABLE_GPU:-"OFF"}
FAISS_ENABLE_CUVS=${FAISS_ENABLE_CUVS:-"OFF"}
FAISS_ENABLE_ROCM=${FAISS_ENABLE_ROCM:-"OFF"}
FAISS_ENABLE_RAFT=${FAISS_ENABLE_RAFT:-"OFF"}
CUVS_VERSION=${CUVS_VERSION:-"25.10.*"}
CMAKE_VERSION=${CMAKE_VERSION:-""}
NVIDIA_EXTRA_INDEX_URL=${NVIDIA_EXTRA_INDEX_URL:-"https://pypi.nvidia.com"}
RAPIDS_WHEEL_DIR=${RAPIDS_WHEEL_DIR:-"/usr/local/rapids-wheel-deps"}
BUILD_PYTHON=${BUILD_PYTHON:-"$(command -v python3)"}
if [ "$FAISS_ENABLE_CUVS" == "ON" ] && [ -x "/opt/python/cp310-cp310/bin/python3" ]; then
    BUILD_PYTHON="/opt/python/cp310-cp310/bin/python3"
fi

find_supported_cuda_host_compiler() {
    local candidate
    local compiler_version
    local compiler_major
    for candidate in /opt/rh/gcc-toolset-13/root/usr/bin/g++ /usr/bin/g++ "$(command -v g++ 2>/dev/null || true)"; do
        if [ -z "$candidate" ] || [ ! -x "$candidate" ]; then
            continue
        fi
        compiler_version=$("$candidate" -dumpfullversion 2>/dev/null || "$candidate" -dumpversion)
        compiler_major=${compiler_version%%.*}
        if [ "$compiler_major" -le 13 ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

apply_cuda_host_compiler() {
    export CUDAHOSTCXX="$1"
    : "${CXX:=$CUDAHOSTCXX}" && export CXX
    local candidate_cc="$(dirname "$CUDAHOSTCXX")/gcc"
    if [ -z "${CC:-}" ] && [ -x "$candidate_cc" ]; then
        export CC="$candidate_cc"
    fi
    printf 'Using CUDA host compiler: %s\n' "$CUDAHOSTCXX"
}

if [ "$FAISS_ENABLE_CUVS" == "ON" ]; then
    if cuda_host_cxx=$(find_supported_cuda_host_compiler); then
        apply_cuda_host_compiler "$cuda_host_cxx"
    else
        "${PKG_INSTALL[@]}" gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
        if [ -x /opt/rh/gcc-toolset-13/root/usr/bin/g++ ]; then
            apply_cuda_host_compiler /opt/rh/gcc-toolset-13/root/usr/bin/g++
        else
            export CMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS:+${CMAKE_CUDA_FLAGS} }-allow-unsupported-compiler"
            printf 'Falling back to CMAKE_CUDA_FLAGS=%s\n' "$CMAKE_CUDA_FLAGS"
        fi
    fi
fi

export PATH="$(dirname "$(readlink -f "$BUILD_PYTHON")"):${PATH}"

if [ -n "$CMAKE_VERSION" ]; then
    uv pip install --system --python "$BUILD_PYTHON" --no-cache "cmake==${CMAKE_VERSION}"
fi

if [ "$FAISS_ENABLE_GPU" == "ON" ]; then
    if [ "$FAISS_ENABLE_ROCM" != "ON" ]; then
        CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 1)
        CUDA_MINOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 2)
        # Install CUDA
        "${PKG_INSTALL[@]}" "${REPO_SETUP_PACKAGE}"
        cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/repos/rhel${REPO_MAJOR_VERSION}/x86_64/cuda-rhel${REPO_MAJOR_VERSION}.repo"
        if [ "$PKG_MANAGER" = "dnf" ]; then
            dnf config-manager --add-repo "$cuda_repo_url"
        else
            yum-config-manager --add-repo "$cuda_repo_url"
        fi
        "${PKG_INSTALL[@]}" cuda-toolkit-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}-${CUDA_VERSION}
        ln -s cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} /usr/local/cuda
        export PATH="/usr/local/cuda/bin:${PATH}"
    fi
fi

if [ "$FAISS_ENABLE_CUVS" == "ON" ]; then
    if [ "$FAISS_ENABLE_GPU" != "ON" ]; then
        echo "FAISS_ENABLE_CUVS requires FAISS_ENABLE_GPU=ON" >&2
        exit 1
    fi

    # libcuvs/libraft wheels declare NCCL as a runtime dependency.
    # Because we install RAPIDS wheels with --no-deps, provide libnccl.so.2 explicitly for the linker.
    mkdir -p "$RAPIDS_WHEEL_DIR"
    uv pip install --python "$BUILD_PYTHON" --target "$RAPIDS_WHEEL_DIR" --no-cache \
        --extra-index-url "${NVIDIA_EXTRA_INDEX_URL}" --only-binary :all: --no-deps \
        "libcuvs-cu12==${CUVS_VERSION}" \
        "libraft-cu12==${CUVS_VERSION}" \
        "librmm-cu12==${CUVS_VERSION}" \
        "rapids-logger==0.1.*" \
        "nvidia-nccl-cu12>=2.19"

    RAPIDS_CMAKE_PREFIX_PATH="${RAPIDS_WHEEL_DIR}/libcuvs;${RAPIDS_WHEEL_DIR}/libraft;${RAPIDS_WHEEL_DIR}/librmm;${RAPIDS_WHEEL_DIR}/rapids_logger"
    export cuvs_DIR="${RAPIDS_WHEEL_DIR}/libcuvs/lib64/cmake/cuvs"
    export raft_DIR="${RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/raft"
    export rmm_DIR="${RAPIDS_WHEEL_DIR}/librmm/lib64/cmake/rmm"
    export hnswlib_DIR="${RAPIDS_WHEEL_DIR}/libcuvs/lib64/cmake/hnswlib"
    export cuco_DIR="${RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/cuco"
    export NvidiaCutlass_DIR="${RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/NvidiaCutlass"
    export nvtx3_DIR="${RAPIDS_WHEEL_DIR}/librmm/lib64/cmake/nvtx3"
    export rapids_logger_DIR="${RAPIDS_WHEEL_DIR}/rapids_logger/lib64/cmake/rapids_logger"
    export CMAKE_PREFIX_PATH="${RAPIDS_CMAKE_PREFIX_PATH}${CMAKE_PREFIX_PATH:+;${CMAKE_PREFIX_PATH}}"
fi

# Build and patch faiss
FAISS_OPT_LEVEL=${FAISS_OPT_LEVEL:-"generic"}
CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-"70-real;80-real"}
NJOB=${NJOB:-"$(nproc --all)"}
cd faiss
cmake_args=(
    -DFAISS_OPT_LEVEL=${FAISS_OPT_LEVEL}
    -DFAISS_ENABLE_GPU=${FAISS_ENABLE_GPU}
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
    -DFAISS_ENABLE_CUVS=${FAISS_ENABLE_CUVS}
    -Dcuvs_DIR=${cuvs_DIR:-}
    -Draft_DIR=${raft_DIR:-}
    -Drmm_DIR=${rmm_DIR:-}
    -Dhnswlib_DIR=${hnswlib_DIR:-}
    -Dcuco_DIR=${cuco_DIR:-}
    -DNvidiaCutlass_DIR=${NvidiaCutlass_DIR:-}
    -Dnvtx3_DIR=${nvtx3_DIR:-}
    -Drapids_logger_DIR=${rapids_logger_DIR:-}
    -DFAISS_ENABLE_RAFT=${FAISS_ENABLE_RAFT}
    -DFAISS_ENABLE_ROCM=${FAISS_ENABLE_ROCM}
    -DFAISS_ENABLE_PYTHON=OFF
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_BUILD_TYPE=Release
)
if [ -n "${CUDAHOSTCXX:-}" ]; then
    cmake_args+=("-DCMAKE_CUDA_HOST_COMPILER=${CUDAHOSTCXX}")
fi
if [ -n "${CC:-}" ]; then
    cmake_args+=("-DCMAKE_C_COMPILER=${CC}")
fi
if [ -n "${CXX:-}" ]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER=${CXX}")
fi
if [ -n "${CMAKE_CUDA_FLAGS:-}" ]; then
    cmake_args+=("-DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
fi
cmake . -B build "${cmake_args[@]}"
cmake --build build -j${NJOB}
cmake --install build
ccache --show-stats || true
cd ..
git apply patch/modify-numpy-find-package.patch
