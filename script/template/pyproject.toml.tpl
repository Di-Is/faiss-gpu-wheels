# Copyright (c) 2024 Di-Is
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# 
# This file is created by the `generate_pyproject.py`. 

[build-system]
requires = ["scikit-build-core", "swig", "oldest-supported-numpy"]
build-backend = "scikit_build_core.build"

[project]
name = "faiss"
version = "1.9.0.post1"
dependencies = ["numpy<2", "packaging"]
requires-python = ">=3.9,<3.13"
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
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only"
]

[project.urls]
faiss-gpu-wheels = "https://github.com/Di-Is/faiss-gpu-wheels"

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
test-requires = [
    "pytest",
    "scipy",
    "torch",
    "pytest-xdist"
]
environment-pass = ["NJOB", "FAISS_OPT_LEVEL"]

[tool.cibuildwheel.linux.environment]
