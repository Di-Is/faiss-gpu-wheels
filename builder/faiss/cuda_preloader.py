"""CUDA shared libraries preloader.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""  # noqa: INP001

from __future__ import annotations

import ctypes
import os
import re
from importlib.metadata import distribution
from logging import getLogger
from pathlib import Path
from typing import TypedDict

logger = getLogger(__name__)


class PackageFileSearchArg(TypedDict):
    """file search query parameter."""

    # python package name
    package_name: str
    # search target filename regex
    filename_regex: str


def search_install_file_path(package_name: str, filename_regex: str) -> str:
    """Search file from python libraries.

    Get absolute paths of files contained in installed python packages assume that
    there is only one file to search for in the Python package to be searched.

    Args:
        package_name: python package name
        filename_regex: search target filename regex

    Returns:
        file absolute path
    """
    for file in distribution(package_name).files:
        if re.search(filename_regex, file.name):
            return str(file.locate())
    msg = f"{filename_regex} pattern file isn't found in {package_name} package"
    raise FileNotFoundError(msg)


class CudaPreLoader:
    """Preload PyPI distributed CUDA shared libraries."""

    cuda_major_file: str = "TARGET_CUDA_MAJOR.txt"

    def __init__(self) -> None:
        """Constructor."""
        cuda_major_file_path = Path(__file__).parent / self.cuda_major_file
        with cuda_major_file_path.open("r") as f:
            self._cuda_major = f.read()
        # load target shared libraries
        self._file_search_targets: list[PackageFileSearchArg] = [
            {
                "package_name": f"nvidia-cuda-runtime-cu{self._cuda_major}",
                "filename_regex": f"libcudart.so.{self._cuda_major}*",
            },
            {
                "package_name": f"nvidia-cublas-cu{self._cuda_major}",
                "filename_regex": f"libcublas.so.{self._cuda_major}",
            },
        ]

    def load(self) -> None:
        """Global load CUDA shared libraries."""
        for target in self._file_search_targets:
            fmt = "Try to load CUDA shared library (regex: {}) in pakcage({})."
            logger.debug(fmt.format(target["filename_regex"], target["package_name"]))

            path = search_install_file_path(**target)
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            fmt = "Finish to load CUDA shared library in {}."
            logger.debug(fmt.format(path))


if not os.getenv("FAISS_MANUALLY_CUDA_PRELOAD"):
    CudaPreLoader().load()
