"""Shared library preloader."""  # noqa: INP001

from __future__ import annotations

import ctypes
import re
from importlib.metadata import distribution
from logging import getLogger
from typing import TypedDict

logger = getLogger(__name__)


class PackageFileSearchArg(TypedDict):
    """file search parameter."""

    # python package name
    package_name: str
    # search target filename regex
    filename_regex: str
    # package group name
    group: str


def search_install_file_path(package_name: str, filename_regex: str) -> str:
    """Search file from python libraries.

    Get absolute paths of files contained in installed python packages assume that
    there is only one file to search for in the Python package to be searched.

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
    msg = f"{filename_regex} pattern file isn't found in {package_name} package"
    raise FileNotFoundError(msg)


def load_shared_library(file_search_target: PackageFileSearchArg) -> None:
    """Global load shared libraries."""
    logger.debug("Try to load shared library in package.", extra=file_search_target)
    path = search_install_file_path(
        file_search_target["package_name"], file_search_target["filename_regex"]
    )
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    logger.debug(
        "Finish to load shared library.",
        extra={"path": path, **file_search_target},
    )
