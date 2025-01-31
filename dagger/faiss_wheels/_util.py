"""Util package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from dagger import CacheVolume, Container, dag

UV_VERSION: str = "0.5.5"
UV_CACHE: CacheVolume = dag.cache_volume("uv-cache")


def install_uv(
    ctr: Container, uv_version: str = UV_VERSION, python_preference: str = "only-system"
) -> Container:
    """Install uv to container.

    Args:
        ctr: container
        uv_version: install target uv version
        python_preference: uv python preference option

    Returns:
        uv installed container
    """
    return (
        ctr.with_file(
            "/usr/local/bin/uv",
            dag.container().from_(f"ghcr.io/astral-sh/uv:{uv_version}").file("/uv"),
        )
        .with_env_variable("UV_PYTHON_PREFERENCE", python_preference)
        .with_env_variable("UV_HTTP_TIMEOUT", "10000000")
        .with_mounted_cache("/root/.cache/uv", UV_CACHE)
        .with_env_variable("UV_LINK_MODE", "copy")
    )


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
