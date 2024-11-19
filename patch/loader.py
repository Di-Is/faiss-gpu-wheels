"""loader.py wapper loading shared libraries.

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


class _PreloadTarget(TypedDict):
    """file search parameter."""

    # python package name
    package: str
    # search target filename regex
    library_regex: str


def _search_install_file_path(package: str, library_regex: str) -> str:
    """Search file from python libraries.

    Get absolute paths of files contained in installed python packages assume that
    there is only one file to search for in the Python package to be searched.

    Args:
        package: python package name
        library_regex: search target filename regex

    Raises:
        FileNotFoundError: raise if the searched file does not exist

    Returns:
        file absolute path
    """
    for file in distribution(package).files:
        if re.search(library_regex, file.name):
            return str(file.locate())
    msg = f"{library_regex} pattern file isn't found in {package} package"
    raise FileNotFoundError(msg)


def _load_shared_library(preload_target: _PreloadTarget) -> None:
    """Global load shared libraries.

    Args:
        preload_target: preload target
    """
    logger.debug("Try to load shared library in package.", extra=preload_target)
    path = _search_install_file_path(preload_target["package"], preload_target["library_regex"])
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    logger.debug("Finish to load shared library.", extra={"path": path, **preload_target})


try:
    import tomllib

    def _load_toml(file_path: Path) -> dict:
        """Load a TOML file using tomli (Python >= 3.11).

        Args:
            file_path: config file path

        Returns:
            loaded config
        """
        with file_path.open("rb") as f:
            return tomllib.load(f)
except ImportError:
    import tomli

    def _load_toml(file_path: Path) -> dict:
        """Load a TOML file using tomli (Python < 3.11).

        Args:
            file_path: config file path

        Returns:
            loaded config
        """
        with file_path.open("rb") as f:
            return tomli.load(f)


_config_path = Path(__file__).parent / "_preload_library.toml"

# loading shared libraries
if _config_path.exists():
    preload_target_groups: dict[str, list[_PreloadTarget]] = _load_toml(_config_path)
else:
    preload_target_groups = {}

for group, targets in preload_target_groups.items():
    if os.getenv(f"_FAISS_WHEEL_DISABLE_{group.upper()}_PRELOAD"):
        logger.debug("Skip to load shared library.", extra={"group": group, "libraries": targets})
    for target in targets:
        _load_shared_library(target)


from ._loader import *  # noqa: E402, F403
