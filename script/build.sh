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
    PKG_INSTALL='dnf install -y'
    REPO_SETUP_PACKAGE='dnf-plugins-core'
    REPO_MAJOR_VERSION=$(rpm -E '%{rhel}')
    MAKE_CACHE='dnf makecache'
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER='yum'
    PKG_INSTALL='yum install -y'
    REPO_SETUP_PACKAGE='yum-utils'
    REPO_MAJOR_VERSION=$(rpm -E '%{rhel}')
    MAKE_CACHE='yum makecache'
else
    echo "Neither dnf nor yum is available." >&2
    exit 1
fi

[ ! -d "$RPM_CACHE_DIR" ] && mkdir -p "$RPM_CACHE_DIR"
eval "$MAKE_CACHE"

# Install blas
eval "$PKG_INSTALL openblas-devel openblas-static"

FAISS_ENABLE_GPU=${FAISS_ENABLE_GPU:-"OFF"}
FAISS_ENABLE_CUVS=${FAISS_ENABLE_CUVS:-"OFF"}
FAISS_ENABLE_ROCM=${FAISS_ENABLE_ROCM:-"OFF"}
FAISS_ENABLE_RAFT=${FAISS_ENABLE_RAFT:-"OFF"}
CUVS_VERSION=${CUVS_VERSION:-"25.10.*"}
CMAKE_VERSION=${CMAKE_VERSION:-""}
NVIDIA_EXTRA_INDEX_URL=${NVIDIA_EXTRA_INDEX_URL:-"https://pypi.nvidia.com"}
BUILD_PYTHON=${BUILD_PYTHON:-"$(command -v python3)"}
if [ "$FAISS_ENABLE_CUVS" == "ON" ] && [ -x "/opt/python/cp310-cp310/bin/python3" ]; then
    BUILD_PYTHON="/opt/python/cp310-cp310/bin/python3"
fi

PYTHON_BIN_DIR=$("$BUILD_PYTHON" - <<'PY'
from pathlib import Path
import sys

print(Path(sys.executable).resolve().parent)
PY
)
export PATH="${PYTHON_BIN_DIR}:${PATH}"

if [ -n "$CMAKE_VERSION" ]; then
    uv pip install --system --python "$BUILD_PYTHON" --no-cache "cmake==${CMAKE_VERSION}"
fi

if [ "$FAISS_ENABLE_GPU" == "ON" ]; then
    if [ "$FAISS_ENABLE_ROCM" != "ON" ]; then
        CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 1)
        CUDA_MINOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 2)
        # Install CUDA
        eval "$PKG_INSTALL ${REPO_SETUP_PACKAGE}"
        if [ "$PKG_MANAGER" = "dnf" ]; then
            dnf config-manager --add-repo \
                "https://developer.download.nvidia.com/compute/cuda/repos/rhel${REPO_MAJOR_VERSION}/x86_64/cuda-rhel${REPO_MAJOR_VERSION}.repo"
            dnf clean expire-cache
            dnf install -y cuda-toolkit-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}-${CUDA_VERSION}
        else
            yum-config-manager --add-repo \
                "https://developer.download.nvidia.com/compute/cuda/repos/rhel${REPO_MAJOR_VERSION}/x86_64/cuda-rhel${REPO_MAJOR_VERSION}.repo"
            yum install -y cuda-toolkit-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}-${CUDA_VERSION}
        fi
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
    uv pip install --system --python "$BUILD_PYTHON" --no-cache \
        --extra-index-url "${NVIDIA_EXTRA_INDEX_URL}" --only-binary :all: --no-deps \
        "libcuvs-cu12==${CUVS_VERSION}" \
        "libraft-cu12==${CUVS_VERSION}" \
        "librmm-cu12==${CUVS_VERSION}" \
        "rapids-logger==0.1.*" \
        "nvidia-nccl-cu12>=2.19"

    RAPIDS_CMAKE_PREFIX_PATH=$("$BUILD_PYTHON" - <<'PY'
from importlib import import_module
from pathlib import Path

modules = ["libcuvs", "libraft", "librmm", "rapids_logger"]
print(";".join(str(Path(import_module(name).__file__).resolve().parent) for name in modules))
PY
)
    export cuvs_DIR=$("$BUILD_PYTHON" - <<'PY'
from importlib import import_module
from pathlib import Path

print(Path(import_module("libcuvs").__file__).resolve().parent / "lib64/cmake/cuvs")
PY
)
    export raft_DIR=$("$BUILD_PYTHON" - <<'PY'
from importlib import import_module
from pathlib import Path

print(Path(import_module("libraft").__file__).resolve().parent / "lib64/cmake/raft")
PY
)
    export rmm_DIR=$("$BUILD_PYTHON" - <<'PY'
from importlib import import_module
from pathlib import Path

print(Path(import_module("librmm").__file__).resolve().parent / "lib64/cmake/rmm")
PY
)
    export rapids_logger_DIR=$("$BUILD_PYTHON" - <<'PY'
from importlib import import_module
from pathlib import Path

print(Path(import_module("rapids_logger").__file__).resolve().parent / "lib64/cmake/rapids_logger")
PY
)
    export CMAKE_PREFIX_PATH="${RAPIDS_CMAKE_PREFIX_PATH}${CMAKE_PREFIX_PATH:+;${CMAKE_PREFIX_PATH}}"
fi

# Build and patch faiss
FAISS_OPT_LEVEL=${FAISS_OPT_LEVEL:-"generic"}
CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-"70-real;80-real"}
NJOB=${NJOB:-"$(nproc --all)"}
cd faiss
cmake . -B build \
    -DFAISS_OPT_LEVEL=${FAISS_OPT_LEVEL} \
    -DFAISS_ENABLE_GPU=${FAISS_ENABLE_GPU} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -DFAISS_ENABLE_CUVS=${FAISS_ENABLE_CUVS} \
    -Dcuvs_DIR=${cuvs_DIR:-} \
    -Draft_DIR=${raft_DIR:-} \
    -Drmm_DIR=${rmm_DIR:-} \
    -Drapids_logger_DIR=${rapids_logger_DIR:-} \
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
