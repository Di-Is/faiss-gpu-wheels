"""Serve test function using dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import Container


async def test_import(container: Container) -> str:
    """Test import.

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(["python3", "-c", "import faiss"])
    return await container.stdout()


async def test_cpu(container: Container) -> str:
    """Test faiss-cpu test.

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(["pytest", "./faiss/tests"])
    return await container.stdout()


async def test_cpu_torch_contlib(container: Container) -> str:
    """Test faiss-cpu torch contlib test.

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(["pytest", "-s", "./faiss/tests/torch_test_contrib.py"])
    return await container.stdout()


async def test_gpu(container: Container) -> str:
    """Test faiss-gpu torch test.

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(
        ["cp", "./faiss/tests/common_faiss_tests.py", "./faiss/faiss/gpu/test/"],
    ).with_exec(["pytest", "./faiss/faiss/gpu/test"])
    return await container.stdout()


async def test_gpu_torch_contlib(container: Container) -> str:
    """Test faiss-gpu torch contlib test.

    Args:
        container: container installed python/faiss

    Returns:
        output string during test
    """
    container = container.with_exec(
        ["cp", "./faiss/tests/common_faiss_tests.py", "./faiss/faiss/gpu/test/"],
    ).with_exec(["pytest", "-s", "./faiss/faiss/gpu/test/torch_test_contrib_gpu.py"])
    return await container.stdout()
