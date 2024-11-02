"""Shared libraries preload trigger."""  # noqa: INP001

from __future__ import annotations

import json
import os
from logging import getLogger
from pathlib import Path

from .shared_library_preloader import PackageFileSearchArg, load_shared_library

logger = getLogger(__name__)
# loading shared libraries
with (Path(__file__).parent / "PRELOAD_SHARED_LIBRARIES.json").open("r") as f:
    file_search_targets: list[PackageFileSearchArg] = json.load(f)


for target in file_search_targets:
    if os.getenv(f"FAISS_MANUALLY_{target['group'].upper()}_PRELOAD"):
        logger.debug("Skip to load shared library.", extra=target)
    load_shared_library(target)
