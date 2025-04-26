"""Static analysis package.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import Container, Directory, dag


def _build_image(source: Directory) -> Container:
    """Build minimum image.

    Returns:
        image mounted repository code.
    """
    return (
        dag.container()
        .from_("jdxcode/mise")
        .with_directory("/project", source)
        .with_workdir("/project")
        .with_exec(["mise", "trust"])
        .with_exec(["mise", "install"])
    )


async def lint(source: Directory) -> str:
    """Linting code.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(["ruff", "check", "."]).stdout()


async def check_python_format(source: Directory) -> str:
    """Formatting python code.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(["ruff", "format", ".", "--diff"]).stdout()


async def check_toml_format(source: Directory) -> str:
    """Formatting toml config.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(["taplo", "format", "--diff", "*", "*/*", "*/*/*"]).stdout()


async def check_typo(source: Directory) -> str:
    """Checking typo.

    Args:
        source: source directory

    Returns:
        stdout at runtime
    """
    ctr = await _build_image(source).sync()
    return await ctr.with_exec(["typos", "."]).stdout()


async def static_analysis(source: Directory) -> None:
    """Execute static analysis.

    Args:
        source: source directory
    """
    await lint(source)
    await check_python_format(source)
    await check_toml_format(source)
    await check_typo(source)
