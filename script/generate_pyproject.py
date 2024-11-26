#!/usr/bin/env -S uv run --script -q # noqa: EXE003
"""Generate pyproject.toml.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "packaging",
#     "click",
#     "pydanclick",
#     "requests",
#     "tomli-w",
# ]
# ///
import json
import subprocess
from pathlib import Path
from typing import Literal

import click
import requests
import tomli_w
import tomllib
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydanclick import from_pydantic
from pydantic import BaseModel


def get_cuda_version(major: int, minor: int, patch: int, component: str) -> str:
    """Get cuda toolkit component version.

    Args:
        major: cuda major version
        minor: cuda minor version
        patch: cuda patch version
        component: component name

    Raises:
        ValueError: _description_

    Returns:
        component version
    """
    json_file = f"version_{major}.{minor}.{patch}.json"

    cache_dir = Path(__file__).parent / ".cache"
    if not cache_dir.exists():
        cache_dir.mkdir()
    cache_file = cache_dir / json_file
    if cache_file.exists():
        with cache_file.open("rb") as f:
            data = json.load(f)
    else:
        url = f"https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/{json_file}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        with cache_file.open("w") as f:
            json.dump(data, f)
    if component not in data:
        msg = f"Not found {component} data"
        raise ValueError(msg)
    return data[component]["version"]


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


class Args(BaseModel):
    """Script argument."""

    variant: Literal["cpu", "gpu-cu11", "gpu-cu12"]


@click.command()
@from_pydantic(Args)
def cli(args: Args) -> None:
    """CLI function."""
    variant_path = Path(__file__).parent.parent / "variant" / f"faiss-{args.variant}"
    with (variant_path / "config.toml").open("rb") as f:
        config = tomllib.load(f)

    with (Path(__file__).parent / "template" / "pyproject.toml.tpl").open("rb") as f:
        pyproject = tomllib.load(f)

    preload_config_path = variant_path / "_preload_library.toml"
    with preload_config_path.open("wb") as f:
        tomli_w.dump({"preload-library": config["python"]["preload-library"]}, f)

    requires_python = config["python"]["requires-python"]
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ] + [f"Programming Language :: Python :: {v}" for v in get_compatible_python(requires_python)]
    dependencies = ["numpy<2", "packaging", "tomli; python_version < '3.11'"]
    optional_dependencies = {}
    match args.variant:
        case "cpu":
            cmake_define = {"FAISS_ENABLE_GPU": "OFF", "FAISS_ENABLE_RAFT": "OFF"}

        case "gpu-cu11" | "gpu-cu12":
            major = config["cuda"]["major"]
            minor = config["cuda"]["minor"]
            patch = config["cuda"]["patch"]
            cublas_version = get_cuda_version(major, minor, patch, "libcublas")
            cudart_version = get_cuda_version(major, minor, patch, "cuda_cudart")
            dependencies += [
                f"nvidia-cuda-runtime-cu{major}>={cudart_version}",
                f"nvidia-cublas-cu{major}>={cublas_version}",
            ]
            optional_dependencies = {
                "fix-cuda": [
                    f"nvidia-cuda-runtime-cu{major}=={cudart_version}",
                    f"nvidia-cublas-cu{major}=={cublas_version}",
                ]
            }
            classifiers += [
                "Environment :: GPU :: NVIDIA CUDA",
                f"Environment :: GPU :: NVIDIA CUDA :: {major}",
            ]
            cmake_define = {"FAISS_ENABLE_GPU": "ON", "FAISS_ENABLE_RAFT": "OFF"}

    pyproject["project"]["name"] = f"faiss-{args.variant}"
    pyproject["project"]["requires-python"] = requires_python
    pyproject["project"]["dependencies"] = dependencies
    pyproject["project"]["optional-dependencies"] = optional_dependencies
    pyproject["project"]["classifiers"] = classifiers
    pyproject["tool"]["scikit-build"]["cmake"]["define"] = cmake_define

    pyproject_path = variant_path / "pyproject.toml"
    text = """# Copyright (c) 2024 Di-Is
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
# This file is created by the `generate_pyproject.py`."""
    pyproject_path.write_text(f"{text}\n" + tomli_w.dumps(pyproject))

    _ = subprocess.run(  # noqa: S603
        ["uv", "run", "taplo", "format", str(pyproject_path)],  # noqa: S607
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    cli()
