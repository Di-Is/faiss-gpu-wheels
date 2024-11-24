"""Static analysis package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import Container, Directory, dag

from ._util import install_uv


def _build_image(source: Directory) -> Container:
    """Build minimum image.

    Returns:
        image mounted repository code.
    """
    return (
        install_uv(dag.container().from_("ubuntu:24.04"), python_preference="only-managed")
        .with_directory("/project", source)
        .with_workdir("/project")
        .with_exec(["uv", "sync"])
    )


async def lint(source: Directory) -> str:
    """Linting code.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(["uv", "run", "ruff", "check", "."]).stdout()


async def check_python_format(source: Directory) -> str:
    """Formatting python code.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(["uv", "run", "ruff", "format", ".", "--diff"]).stdout()


async def check_toml_format(source: Directory) -> str:
    """Formatting toml config.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(
        ["uv", "run", "taplo", "format", "--diff", "*", "*/*", "*/*/*"]
    ).stdout()


async def check_typo(source: Directory) -> str:
    """Checking typo.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(["uv", "run", "typos", "."]).stdout()


async def static_analysis(source: Directory) -> None:
    """Execute static analysis.

    Args:
        source: source directory
    """
    await lint(source)
    await check_python_format(source)
    await check_toml_format(source)
    await check_typo(source)
