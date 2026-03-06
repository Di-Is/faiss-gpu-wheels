# faiss-wheels

This repository is based on [kyamagu/faiss-wheels](https://github.com/kyamagu/faiss-wheels).

[![PyPI](https://img.shields.io/pypi/v/faiss-gpu-cu11?label=faiss-gpu-cu11)](https://pypi.org/project/faiss-gpu-cu11/)
[![PyPI](https://img.shields.io/pypi/v/faiss-gpu-cu12?label=faiss-gpu-cu12)](https://pypi.org/project/faiss-gpu-cu12/)

## Overview

This repository provides scripts to build GPU-enabled wheels for the [faiss](https://github.com/facebookresearch/faiss) library.
Distributes `faiss-gpu-cuXX` and `faiss-gpu-cuvs` packages to PyPI using the contents of this repository.

### Key Features

* **No local CUDA installation required** - Dynamically links to CUDA Runtime and cuBLAS libraries from PyPI
* Builds CUDA 11.8+, CUDA 12.1+, and CUDA 12.4 + cuVS compatible wheels
* Supports Volta to Ada Lovelace architecture GPUs (Compute Capability 7.0–8.9)
* Bundles OpenBLAS in Linux
* Reduces wheel file size through dynamic linking instead of static compilation
* Supports cuVS builds by dynamically loading `libcuvs` from RAPIDS wheels

## Important Requirements

The published `faiss-gpu-cuXX` packages require proper system setup that cannot be managed by pip. It is your responsibility to prepare a suitable environment:

1. **NVIDIA Driver**: Your host must have a CUDA-compatible NVIDIA driver installed
   * The minimum driver version depends on the CUDA version that gets installed
   * NVIDIA drivers are backward compatible with older CUDA versions ([See CUDA Compatibility Documentation](https://docs.nvidia.com/deploy/cuda-compatibility/))

2. **GPU Architecture**: Your GPU must be compatible (Compute Capability 7.0–8.9)
   * Supported: Volta, Turing, Ampere, Ada Lovelace

3. **Library Compatibility**: If you install multiple CUDA-dependent libraries (e.g., PyTorch) in the same environment, they must link to the same CUDA version

## GPU Architecture Support for PyPI Packages

### Support Policy for `faiss-gpu-cu11`, `faiss-gpu-cu12`, and `faiss-gpu-cuvs`

**Note**: This is an **unofficial, personal development project** with limited computational resources. Due to these constraints, comprehensive testing across all NVIDIA GPU architectures is not feasible. The pre-built GPU packages on PyPI aim to support the same GPU architecture range (Compute Capability 7.0–8.9) as the official Faiss repository.

### Sponsoring New GPU Architecture Support

Adding support for a new GPU architecture (e.g., Hopper, Blackwell) requires dedicated hardware for building and testing. NVIDIA GPUs have limited compatibility across compute capabilities — binaries built for one architecture do not necessarily work correctly on another. Distributing untested wheels is not an option.

This is an unfunded personal project. If you or your organization need support for an architecture outside the current range, please consider [sponsoring this project](https://github.com/sponsors/Di-Is) to help cover the hardware and infrastructure costs. For ongoing discussion and status updates, see [Support for New GPU Architectures](https://github.com/Di-Is/faiss-gpu-wheels/discussions/120).

### For Unsupported GPU Architectures

If you have a GPU architecture that is not supported by these pre-built wheels:

1. **Official Faiss**: Follow the [official Faiss repository build instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
2. **Build from Source**: Use this repository's code to build wheels for your specific architecture (see [Building from Source](#building-from-source) section)

## Installation

The `faiss-gpu-cu11`, `faiss-gpu-cu12`, and `faiss-gpu-cuvs` wheels are available on PyPI. Choose the appropriate version for your CUDA environment.

### For CUDA 12

```bash
# Install with fixed CUDA 12.1 (requires NVIDIA Driver ≥R530)
pip install 'faiss-gpu-cu12[fix-cuda]'

# Install with CUDA 12.X (X≥1) - allows flexibility but driver requirement varies
pip install faiss-gpu-cu12
```

**Details:**

* `faiss-gpu-cu12` is built with CUDA Toolkit 12.1 and maintains minor version compatibility
* With `[fix-cuda]`: Installs exactly CUDA 12.1, requiring NVIDIA Driver ≥R530
* Without `[fix-cuda]`: Allows any CUDA 12.X (X≥1), driver requirement depends on the actual CUDA version installed
  * For example: CUDA 12.4 requires Driver ≥R550
* Use without `[fix-cuda]` when integrating with other CUDA-dependent packages (e.g., PyTorch with CUDA 12.4)

**System Requirements:**

* OS: Linux x86_64 (glibc ≥2.17)
* GPU: Compute Capability 7.0–8.9

### For CUDA 12.4 + cuVS

```bash
# Install with fixed CUDA 12.4 and RAPIDS cuVS runtime
pip install 'faiss-gpu-cuvs[fix-cuda]' --extra-index-url=https://pypi.nvidia.com

# Same libcuvs series, but allow CUDA 12.X runtime packages to float
pip install faiss-gpu-cuvs --extra-index-url=https://pypi.nvidia.com
```

**Details:**

* `faiss-gpu-cuvs` is built with CUDA Toolkit 12.4.1 and links against `libcuvs-cu12==25.10.*`
* The build and runtime `libcuvs` series are intentionally pinned to the same RAPIDS release line
* The wheel is built with a `manylinux_2_28` base image so that the advertised libc baseline matches the cuVS-enabled build
* `[fix-cuda]` pins CUDA runtime, cuBLAS, cuRAND, cuSOLVER, cuSPARSE, and nvJitLink to the build baseline while keeping the same cuVS line
* Installation requires the NVIDIA package index because `libcuvs-cu12` and related RAPIDS binary wheels are distributed there
* At import time, the wheel delegates cuVS loading to `libcuvs.load_library()`, which loads `libcuvs`, `libraft`, `librmm`, and the required CUDA-side dependencies in the expected order

**System Requirements:**

* OS: Linux x86_64 (glibc ≥2.28)
* GPU: Compute Capability 7.0–8.9

### For CUDA 11

```bash
# Install with CUDA 11.8 (requires NVIDIA Driver ≥R520)
pip install faiss-gpu-cu11[fix-cuda]

# Same as above (CUDA 11.8 is the final version)
pip install faiss-gpu-cu11
```

**Details:**

* `faiss-gpu-cu11` is built with CUDA Toolkit 11.8
* Both commands install CUDA 11.8 since no newer CUDA 11.X versions exist
* Requires NVIDIA Driver ≥R520

**System Requirements:**

* OS: Linux x86_64 (glibc ≥2.17)
* GPU: Compute Capability 7.0–8.9

### Driver Compatibility Reference

| CUDA Version | Minimum Driver Version |
|--------------|------------------------|
| CUDA 11.8    | ≥R520 (520.61.05)      |
| CUDA 12.1    | ≥R530 (530.30.02)      |
| CUDA 12.4    | ≥R550                  |
| CUDA 12.2+   | Check [NVIDIA Documentation](https://docs.nvidia.com/deploy/cuda-compatibility/) |

**Warning**: When installing without `[fix-cuda]`, pip may resolve to a newer CUDA version that requires a newer driver than you have installed. Always verify driver compatibility before installation.

### Advanced: Using System CUDA Libraries

If you need to use system-installed CUDA instead of PyPI CUDA packages, you can bypass the automatic CUDA loading:

1. **Exclude PyPI CUDA dependencies** using your package manager (e.g., [uv](https://github.com/astral-sh/uv/issues/7214), [pdm](https://pdm-project.org/en/latest/usage/config/#exclude-specific-packages-and-their-dependencies-from-the-lock-file))
2. **Set environment variable**: `_FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1`
3. **Ensure CUDA libraries are accessible** via `LD_LIBRARY_PATH`

Example with uv (workaround):

```toml
# In pyproject.toml
[tool.uv]
override-dependencies = [
    "nvidia-cuda-runtime-cu11==0.0.0; sys_platform == 'never'",
    "nvidia-cublas-cu11==0.0.0; sys_platform == 'never'",
]
```

## Versioning

* Follows the original faiss repository versioning (e.g., `1.11.0`)
* Patches specific to this repository use `postN` suffix (e.g., `1.11.0.post1`)

## Building from Source

Build `faiss-gpu-cu11`, `faiss-gpu-cu12`, and `faiss-gpu-cuvs` wheels using [cibuildwheel](https://github.com/pypa/cibuildwheel).

### Build Configuration

```bash
# Configure build parameters
export NJOB="32"                          # Number of parallel build jobs
export FAISS_OPT_LEVEL="generic"          # Options: generic, avx2, avx512
export CUDA_ARCHITECTURES="70-real;80-real"  # Target GPU architectures

# For builds without GPU testing
export CIBW_TEST_COMMAND_LINUX=""

# For builds with GPU testing (requires NVIDIA Docker)
export CIBW_CONTAINER_ENGINE='docker; create_args: --gpus all'
# Note: GPU testing requires Docker with NVIDIA Container Toolkit configured
```

### Build Commands

```bash
# Build faiss-gpu-cu11 wheels
uvx cibuildwheel@2.23.2 variant/gpu-cu11 --output-dir wheelhouse/gpu-cu11

# Build faiss-gpu-cu12 wheels
uvx cibuildwheel@2.23.2 variant/gpu-cu12 --output-dir wheelhouse/gpu-cu12

# Build faiss-gpu-cuvs wheels
uvx cibuildwheel@2.23.2 variant/gpu-cuvs --output-dir wheelhouse/gpu-cuvs
```

Wheels will be created in `{repository_root}/wheelhouse/gpu-cuXX/`.

### Build Requirements

* OS: Linux x86_64
* NVIDIA Container Toolkit (if running tests)
* NVIDIA Driver: ≥R530 (if running tests with CUDA 12)
* `uv` available in the build environment
* For `faiss-gpu-cuvs`, access to `https://pypi.nvidia.com` to install RAPIDS binary wheels (`libcuvs-cu12`, `libraft-cu12`, `librmm-cu12`, `rapids-logger`)
* For `faiss-gpu-cuvs`, `cmake >= 3.30.4`
