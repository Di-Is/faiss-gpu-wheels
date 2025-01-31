"""loader.py wapper loading shared libraries.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""  # noqa: INP001

from __future__ import annotations

import ctypes
import json
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


def _load_shared_library(preload_target: _PreloadTarget) -> None:
    """Global load shared libraries.

    Args:
        preload_target: preload target
    """
    logger.debug("Try to load shared library in package.", extra=preload_target)
    for file in distribution(preload_target["package"]).files:
        if preload_target["library"] == file.name:
            path = str(file.locate())
            break
    else:
        msg = f"{preload_target['library']} file isn't found in {preload_target['package']}"
        raise FileNotFoundError(msg)
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    logger.debug("Finish to load shared library.", extra=preload_target)


_CONFIG_PATH = Path(__file__).parent / "_preload_library.json"

# loading shared libraries
if _CONFIG_PATH.exists():
    preload_targets: list[_PreloadTarget] = json.loads(_CONFIG_PATH.read_text())["preload-library"]
else:
    preload_targets = []

for target in preload_targets:
    if os.getenv(f"_FAISS_WHEEL_DISABLE_{target['group'].upper()}_PRELOAD"):
        logger.debug("Skip to load shared library.", extra=target)
        continue
    _load_shared_library(target)


from ._loader import *  # noqa: E402, F403
