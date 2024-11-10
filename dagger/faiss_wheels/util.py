"""Util package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import Container, dag


async def install_uv(ctr: Container, uv_version: str) -> Container:
    """Install uv to container.

    Args:
        ctr: container
        uv_version: install target uv version

    Returns:
        uv installed container
    """
    ctr = ctr.with_file(
        "/usr/local/bin/uv",
        dag.container().from_(f"ghcr.io/astral-sh/uv:{uv_version}").file("/uv"),
    )
    return await ctr.sync()
