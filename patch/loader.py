"""loader.py wapper loading shared libraries.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""  # noqa: INP001

from __future__ import annotations

import ctypes
import os
from importlib.metadata import distribution
from logging import getLogger
from pathlib import Path
from typing import TypedDict

logger = getLogger(__name__)


class _PreloadTarget(TypedDict):
    """file search parameter."""

    # package group
    group: str
    # python package name
    package: str
    # search target filename
    library: str


def _search_install_file_path(package: str, library: str) -> str:
    """Search file from python libraries.

    Get absolute paths of files contained in installed python packages assume that
    there is only one file to search for in the Python package to be searched.

    Args:
        package: python package name
        library: search target filename regex

    Raises:
        FileNotFoundError: raise if the searched file does not exist

    Returns:
        file absolute path
    """
    for file in distribution(package).files:
        if library == file.name:
            return str(file.locate())
    msg = f"{library} file isn't found in {package} package"
    raise FileNotFoundError(msg)


def _load_shared_library(preload_target: _PreloadTarget) -> None:
    """Global load shared libraries.

    Args:
        preload_target: preload target
    """
    logger.debug("Try to load shared library in package.", extra=preload_target)
    path = _search_install_file_path(preload_target["package"], preload_target["library"])
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
    preload_targets: list[_PreloadTarget] = _load_toml(_config_path)["preload-library"]
else:
    preload_targets = []

for target in preload_targets:
    if os.getenv(f"_FAISS_WHEEL_DISABLE_{target['group'].upper()}_PRELOAD"):
        logger.debug("Skip to load shared library.", extra=target)
        continue
    _load_shared_library(target)


from ._loader import *  # noqa: E402, F403
