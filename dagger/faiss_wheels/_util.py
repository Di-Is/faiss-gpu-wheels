"""Util package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import CacheVolume, Container, dag

UV_VERSION: str = "0.5.1"
UV_CACHE: CacheVolume = dag.cache_volume("uv-cache")


def install_uv(ctr: Container, uv_version: str) -> Container:
    """Install uv to container.

    Args:
        ctr: container
        uv_version: install target uv version

    Returns:
        uv installed container
    """
    return ctr.with_file(
        "/usr/local/bin/uv",
        dag.container().from_(f"ghcr.io/astral-sh/uv:{uv_version}").file("/uv"),
    )
