# faiss-wheels

This repository is based on [kyamagu/faiss-wheels](https://github.com/kyamagu/faiss-wheels).

[![PyPI](https://img.shields.io/pypi/v/faiss-gpu-cu11?label=faiss-gpu-cu11)](https://pypi.org/project/faiss-gpu-cu11/)
[![PyPI](https://img.shields.io/pypi/v/faiss-gpu-cu12?label=faiss-gpu-cu12)](https://pypi.org/project/faiss-gpu-cu12/)

## Overview

This repository provides scripts to build gpu wheels for the [faiss](https://github.com/facebookresearch/faiss) library.
Distribute the `faiss-gpu-cuXX` package to PyPI using the contents of this repository. 

* Builds CUDA 11.8+/CUDA 12.1+ compatible wheels.
  * Support Pascal\~Hopper architecture GPU (Compute Capability 6.0\~9.0).
  * **Dynamically linked to CUDA Runtime and cuBLAS libraries published in PyPI.**
* Bundles OpenBLAS in Linux.


## Installation

The `faiss-gpu-cu11` and `faiss-gpu-cu12` wheels built for CUDA11 and CUDA12 are available on PyPI.
Install one or the other depending on your environment.
These wheels dynamically link to the CUDA Runtime and cuBLAS shared libraries. This approach helps to reduce the file size of the wheels.

`faiss-gpu-cuXX(XX=11 or 12)` has dependencies on CUDA Runtime (`nvidia-cuda-runtime-cuXX`) and cuBLAS (`nvidia-cublas-cuXX`) released by PyPI, and links shared libraries in these packages. 
**Therefore, there is no need to install CUDA on your host(system).**

### <span style="color: red; ">Caution</span>

The published `faiss-gpu-cuXX` package requires proper setup of system, hardware, and other dependencies that cannot be managed by the package manager (e.g. pip).
It is the responsibility of the user of this package to prepare an environment suitable for its operation.

Here are the main requirements that such an environment should meet (Other conditions may be hidden.)

1. the host environment must have a CUDA-compatible Nvidia Driver installed, as required by `faiss-gpu-cuXX` (see below for details)
2. the GPU architecture of the execution environment must be compatible with `faiss-gpu-cuXX` (see below for details)
3. if you install `faiss-gpu-cuXX` and another library (e.g. Pytorch) that uses dynamically linked CUDA in the same environment, they must be linked to the same CUDA shared library.

### Wheel for CUDA12

`faiss-gpu-cu12` is a package built using CUDA Toolkit 12.1.
The following command will install faiss and the CUDA Runtime and cuBLAS for CUDA 12.1 used at build time.

```bash
# install CUDA 12.1 at the same time
pip install faiss-gpu-cu12[fix_cuda]
```

**Requirements**
* OS: Linux
  * arch: x86_64
  * glibc >=2.28
* Nvidia driver: >=R530 (specify `fix_cuda` extra during installation)
* GPU architectures: Pascal\~Hopper (Compute Capability: 6.0\~9.0)

**Advanced**

The `faiss-gpu-cu12` package (the binaries contained in it) is minor version compatible with CUDA and will work dynamically linked with CUDA 12.X (X>=1).

Installation of the CUDA runtime and cuBLAS is allowed to the extent that minor version compatibility is maintained by excluding the `fix_cuda` extra.

This is useful when coexisting this package with a package that has a dependency on the CUDA Toolkit used at build time, such as Pytorch or Tensorflow.

The installation commands are as follows.

```bash
# install CUDA 12.X(X>=1) at the same time
pip install faiss-gpu-cu12
```

If you install the `faiss-gpu-cuXX` package in this way, CUDA may be updated due to lock file updates, etc.

Please note that this may cause an error depending on the compatibility with the driver. (Basically, to use a new CUDA, the driver must also be updated).


### Wheel for CUDA11

`faiss-gpu-cu11` is a package built using CUDA Toolkit 11.8.
The following command will install faiss and the CUDA Runtime and cuBLAS for CUDA 11.8 used at build time.

```bash
# install CUDA 11.8 at the same time
pip install faiss-gpu-cu11[fix_cuda]
```

**Requirements**
* OS: Linux
  * arch: x86_64
  * glibc >=2.28
* Nvidia driver: >=R520 (specify `fix_cuda` extra during installation)
* GPU architectures: Pascal\~Hopper (Compute Capability: 6.0\~9.0)

**Advanced**

Since CUDA 11.8 is the final version of the CUDA 11 series, the results are the same with or without the `fix_cuda` extras.

```bash
# install CUDA 11.X(X>=8) at the same time
pip install faiss-gpu-cu11
```

### Versioning rule

Packages to be published from this repository are "{A}.{B}.{C}.{D}" format.
A, B, and C are the versions of faiss used for builds.
D is the version used to manage changes specific to this repository.

## Usage

### Build wheels

You can build `faiss-gpu-cu11` and `faiss-gpu-cu12` wheels using [dagger](https://dagger.io).

```bash
# build wheel for CUDA 11.8
dagger call build-gpu-wheels --cuda-major 11 --host-directory=.:build-view --output ./wheelhouse/gpu/cuda11/

# build wheel for CUDA 12.1
dagger call build-gpu-wheels --cuda-major 12 --host-directory=.:build-view --output ./wheelhouse/gpu/cuda12/
```

When executed, a wheel is created under "{repository root}/wheelhouse/gpu/cuXX".


**Requirements**
* OS: Linux
  * arch: x86_64
* Dagger: v0.11.2


### Test wheels

You can test `faiss-gpu-cu11` and `faiss-gpu-cu12` wheels using [dagger](https://dagger.io).


```bash
# test for faiss-gpu-cu11 wheels
_EXPERIMENTAL_DAGGER_GPU_SUPPORT=1 dagger call test-gpu-wheels --cuda-major 11 --host-directory=.:test-view

# test for faiss-gpu-cu12 wheels
_EXPERIMENTAL_DAGGER_GPU_SUPPORT=1 dagger call test-gpu-wheels --cuda-major 12 --host-directory=.:test-view
```

**Requirements**
* OS: Linux
  * arch: x86_64
* Dagger: v0.11.2
* Nvidia container toolkit
* Nvidia driver: >=R530
