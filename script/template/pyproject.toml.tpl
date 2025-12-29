# Copyright (c) 2024 Di-Is
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# 
# This file is created by the `generate_pyproject.py`. 

[build-system]
requires = ["scikit-build-core", "swig", "numpy>=2,<3"]
build-backend = "scikit_build_core.build"

[project]
name = "faiss"
version = "1.13.2"
dependencies = ["numpy>=2,<3", "packaging"]
requires-python = ">=3.10,<3.14"
authors = [{ name = "Di-Is", email = "rhoxbox@gmail.com" }]
description = "A library for efficient similarity search and clustering of dense vectors."
license = { file = "../../LICENSE" }
readme = "../../README.md"
keywords = ["search nearest neighbors"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3 :: Only"
]

[project.urls]
faiss-gpu-wheels = "https://github.com/Di-Is/faiss-gpu-wheels"

[dependency-groups]

[tool.uv]

[tool.scikit-build]
cmake.source-dir = "../../"
wheel.exclude = ["*.cxx"]

[tool.scikit-build.cmake.define]
FAISS_OPT_LEVEL = { env = "FAISS_OPT_LEVEL" }
FAISS_ENABLE_GPU = { env = "FAISS_ENABLE_GPU" }
FAISS_ENABLE_ROCM = { env = "FAISS_ENABLE_ROCM" }
FAISS_ENABLE_RAFT = { env = "FAISS_ENABLE_RAFT" }

[tool.cibuildwheel]
build-frontend = "build[uv]"
skip = "pp* *-musllinux* *i686"

[tool.cibuildwheel.linux]
before-all = "bash script/build.sh"
environment-pass = ["NJOB", "FAISS_OPT_LEVEL"]

[tool.cibuildwheel.linux.environment]
UV_CACHE_DIR='/host/tmp/.cache/uv'
UV_LINK_MODE='copy'
OMP_NUM_THREADS='1'
