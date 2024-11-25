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
version = "1.9.0.0"
authors = [{ name = "Di-Is", email = "rhoxbox@gmail.com" }]
description = "A library for efficient similarity search and clustering of dense vectors."
license = { file = "../../LICENSE" }
readme = "../../README.md"
keywords = ["search nearest neighbors"]

[project.urls]
faiss-gpu-wheels = "https://github.com/Di-Is/faiss-gpu-wheels"

[tool.scikit-build]
cmake.source-dir = "../../"
wheel.exclude = ["*.cxx"]
