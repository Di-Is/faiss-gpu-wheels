"""Serve test function using dagger pipeline

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import Container


async def test_import(container: Container) -> str:
    """test import

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(["python3", "-c", "import faiss"])
    return await container.stdout()


async def test_cpu(container: Container) -> str:
    """test faiss-cpu test

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(
        ["pytest", "./faiss/tests"],
    )
    return await container.stdout()


async def test_cpu_torch_contlib(container: Container) -> str:
    """test faiss-cpu torch contlib test

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(
        ["pytest", "-s", "./faiss/tests/torch_test_contrib.py"],
    )
    return await container.stdout()


async def test_gpu(container: Container) -> str:
    """test faiss-gpu torch test

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(
        [
            "cp",
            "./faiss/tests/common_faiss_tests.py",
            "./faiss/faiss/gpu/test/",
        ],
    ).with_exec(["pytest", "./faiss/faiss/gpu/test"])
    return await container.stdout()


async def test_gpu_torch_contlib(container: Container) -> str:
    """test faiss-gpu torch contlib test

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(
        [
            "cp",
            "./faiss/tests/common_faiss_tests.py",
            "./faiss/faiss/gpu/test/",
        ],
    ).with_exec(["pytest", "-s", "./faiss/faiss/gpu/test/torch_test_contrib_gpu.py"])
    return await container.stdout()


async def install_uv(container: Container) -> Container:
    """install uv to container

    Args:
        container: container

    Returns:
        uv installed container
    """
    # get python path
    python_prefix = (
        await container.with_exec(
            ["python3", "-c", "import sys;print(sys.prefix)"]
        ).stdout()
    ).replace("\n", "")

    # set python directory
    container = container.with_env_variable("VIRTUAL_ENV", python_prefix)

    # check exist
    if len(await container.directory(f"{python_prefix}/bin").glob("python")) == 0:
        container = container.with_exec(
            [
                "/bin/sh",
                "-c",
                f"ln -s `command -v python3` {python_prefix}/bin/python",
            ]
        )

    container = await container.with_exec(
        [
            "bash",
            "-c",
            "yum install -y curl || apt update && apt install -y curl || dnf install -y curl || true",
        ]
    )

    # install ul
    container = container.with_exec(
        ["/bin/sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]
    )

    # set uv path
    uv_path = (
        await container.with_exec(["/bin/sh", "-c", "ls -d ~/.cargo/bin/"]).stdout()
    ).replace("\n", "")
    container = container.with_env_variable("PATH", f"{uv_path}:$PATH", expand=True)
    await container.sync()

    return container
