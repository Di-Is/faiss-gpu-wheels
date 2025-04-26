"""Util package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from dagger import CacheVolume, dag

UV_CACHE: CacheVolume = dag.cache_volume("uv-cache")


def get_compatible_python(requires_python: str) -> list[str]:
    """Get compatible python versions.

    Args:
        requires_python: requires-python format str in pyproject.toml

    Returns:
        compatible python list
    """
    specifier = SpecifierSet(requires_python)
    compatible_versions = []
    minor = 0
    start = False
    while True:
        version = Version(f"3.{minor}")
        if version in specifier:
            compatible_versions.append(str(version))
            minor += 1
            start = True
        elif start:
            break
        else:
            minor += 1
    return compatible_versions
