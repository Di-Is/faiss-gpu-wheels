import os
import ctypes
import re
from typing import List, TypedDict
from importlib.metadata import distribution

from logging import getLogger

logger = getLogger(__name__)


class PackageFileSearchArg(TypedDict):
    # python package name
    package_name: str
    # search target filename regex
    filename_regex: str


def search_install_file_path(package_name: str, filename_regex: str) -> str:
    """Get absolute paths of files contained in installed python packages
    Assume that there is only one file to search for in the Python package to be searched

    Args:
        package_name: python package name
        filename_regex: search target filename regex

    Raises:
        FileNotFoundError: raise if the searched file does not exist

    Returns:
        file absolute path
    """
    for file in distribution(package_name).files:
        if re.search(filename_regex, file.name):
            return str(file.locate())
    else:
        raise FileNotFoundError(
            f"{filename_regex} pattern file isn't found in {package_name} package"
        )


class CudaPreLoader:
    """Preload PyPI distributed CUDA shared libraries"""

    cuda_major_file: str = "TARGET_CUDA_MAJOR.txt"

    def __init__(self):
        """constructor"""
        cuda_major_file = os.path.join(os.path.dirname(__file__), self.cuda_major_file)
        with open(cuda_major_file, "r") as f:
            self._cuda_major = f.read()
        # load target shared libraries
        self._file_search_targets: List[PackageFileSearchArg] = [
            {
                "package_name": f"nvidia-cuda-runtime-cu{self._cuda_major}",
                "filename_regex": f"libcudart.so.{self._cuda_major}*",
            },
            {
                "package_name": f"nvidia-cublas-cu{self._cuda_major}",
                "filename_regex": f"libcublas.so.{self._cuda_major}",
            },
        ]

    def load(self):
        """Global load CUDA shared libraries"""
        for target in self._file_search_targets:
            logger.debug(
                f"Try to load CUDA shared library (regex: {target['filename_regex']}) in pakcage({target['package_name']})."
            )
            path = search_install_file_path(**target)
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            logger.debug(f"Finish to load CUDA shared library in {path}.")



if not os.getenv("FAISS_MANUALLY_CUDA_PRELOAD"):
    CudaPreLoader().load()
