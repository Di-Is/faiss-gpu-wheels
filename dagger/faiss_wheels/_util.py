"""Util package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from typing import Callable

import anyio
import tomllib
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from dagger import CacheVolume, Container, File, dag

from ._type import CpuConfig, GpuCudaConfig

UV_VERSION: str = "0.5.4"
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


async def execute_async_parallel(
    function: Callable, kwargs_list: list[dict], njob: int | None = None
) -> list:
    """Execute async function parallel.

    Args:
        function: async function
        kwargs_list: function kwargs format arguments.
        njob: number of job.

    Returns:
        jon result
    """

    async def _core_function(func: Callable, kwargs_list: list[dict], njob: int = 1) -> list:
        limiter = anyio.CapacityLimiter(njob)
        results = []

        async def _worker(kwargs: dict) -> None:
            async with limiter:
                result = await func(**kwargs)
                results.append(result)

        async with anyio.create_task_group() as tg:
            for kwargs in kwargs_list:
                tg.start_soon(_worker, kwargs)

        return results

    if njob and njob > 1:
        results = await _core_function(function, kwargs_list, njob)
    else:
        results = [await function(**kwargs) for kwargs in kwargs_list]
    return results


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


async def load_config(config: File) -> CpuConfig | GpuCudaConfig:
    """Load config.

    Returns:
        config
    """
    raw_config = tomllib.loads(await config.contents())
    match raw_config["variant"]:
        case "cpu":
            config = CpuConfig(**raw_config)
        case "gpu-cu11" | "gpu-cu12":
            config = GpuCudaConfig(**raw_config)
        case _:
            raise NotImplementedError
    return config
