"""Static analysis package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import Container, Directory, dag

from ._util import UV_CACHE, UV_VERSION

_RUFF_VERSION = "0.7.4"


def _build_image(source: Directory) -> Container:
    """Build minimum image.

    Returns:
        image mounted repository code.
    """
    return (
        dag.container()
        .from_(f"ghcr.io/astral-sh/uv:{UV_VERSION}-debian-slim")
        .with_directory("/project", source)
        .with_workdir("/project")
        .with_mounted_cache("/root/.cache/uv", UV_CACHE)
        .with_env_variable("UV_LINK_MODE", "copy")
    )


async def lint(source: Directory) -> str:
    """Linting code.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await (
        _build_image(source).with_exec(["uv", "tool", "install", f"ruff@{_RUFF_VERSION}"]).sync()
    )
    return await ctr.with_exec(["uvx", "ruff", "check", "."]).stdout()


async def check_format(source: Directory) -> str:
    """Formatting code.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await (
        _build_image(source).with_exec(["uv", "tool", "install", f"ruff@{_RUFF_VERSION}"]).sync()
    )
    return await ctr.with_exec(["uvx", "ruff", "format", ".", "--diff"]).stdout()


async def check_typo(source: Directory) -> str:
    """Checking typo.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).with_exec(["uv", "tool", "install", "typos"]).sync()
    return await ctr.with_exec(["uvx", "typos", "."]).stdout()


async def static_analysis(source: Directory) -> None:
    """Execute static analysis.

    Args:
        source: source directory
    """
    await lint(source)
    await check_format(source)
    await check_typo(source)
