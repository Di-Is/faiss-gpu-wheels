#!/usr/bin/env -S uv run --script -q # noqa: EXE003
"""Generate pyproject.toml.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

# /// script
# requires-python = "~=3.12.0"
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
from pydanclick import from_pydantic
from pydantic import BaseModel


def get_cuda_version(version: str, component: str) -> str:
    """Get cuda toolkit component version.

    Args:
        version: cuda version (x.y.z format)
        component: component name

    Raises:
        ValueError: _description_

    Returns:
        component version
    """
    json_file = f"version_{version}.json"

    # make cache
    cache_dir = Path(__file__).parent / ".cache"
    if not cache_dir.exists():
        cache_dir.mkdir()

    # load cache
    cache_file = cache_dir / json_file
    if cache_file.exists():
        with cache_file.open("rb") as f:
            data = json.load(f)
    else:
        url = f"https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/{json_file}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        # save cache
        with cache_file.open("w") as f:
            json.dump(data, f, indent=4)
    if component not in data:
        msg = f"Not found {component} data"
        raise ValueError(msg)
    return data[component]["version"]


class Args(BaseModel):
    """Script argument."""

    variant: Literal["cpu", "gpu-cu11", "gpu-cu12"]


@click.command()
@from_pydantic(Args)
def cli(args: Args) -> None:
    """CLI function."""
    variant = args.variant
    # load pyproject source config
    variant_path = Path(__file__).parent.parent / "variant" / f"{variant}"
    with (variant_path / "config.toml").open("rb") as f:
        config = tomllib.load(f)

    # load pyproject template
    with (Path(__file__).parent / "template" / "pyproject.toml.tpl").open("rb") as f:
        pyproject = tomllib.load(f)

    # save preload-library list
    preload_config_path = variant_path / "_preload_library.json"
    with preload_config_path.open("w") as f:
        json.dump({"preload-library": config["python"]["preload-library"]}, f, indent=4)

    if variant == "cpu":
        dependencies = []
        optional_dependencies = {}
        classifiers = []
        build_envs = {}
        envs = {}
        test_requires = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
        enviromnet_pass = []
        test_command = """
# CPU Test
pytest {project}/faiss/tests/ -n $((`nproc --all`/5+1)) &&
pytest {project}/faiss/tests/torch_test_contrib.py -n $((`nproc --all`/5+1))
"""
    elif variant in ["gpu-cu11", "gpu-cu12"]:
        major, _, _ = config["cuda"]["version"].split(".")
        cublas_ver = get_cuda_version(config["cuda"]["version"], "libcublas")
        cudart_ver = get_cuda_version(config["cuda"]["version"], "cuda_cudart")
        dependencies = [
            f"nvidia-cuda-runtime-cu{major}>={cudart_ver}",
            f"nvidia-cublas-cu{major}>={cublas_ver}",
        ]
        optional_dependencies = {
            "fix-cuda": [
                f"nvidia-cuda-runtime-cu{major}=={cudart_ver}",
                f"nvidia-cublas-cu{major}=={cublas_ver}",
            ]
        }
        classifiers = [
            "Environment :: GPU :: NVIDIA CUDA",
            f"Environment :: GPU :: NVIDIA CUDA :: {major}",
        ]
        build_envs = {"CUDA_VERSION": config["cuda"]["version"]}
        envs = {"FAISS_ENABLE_GPU": "ON"}
        enviromnet_pass = ["CUDA_ARCHITECTURES"]
        test_command = """
# CPU Test
pytest {project}/faiss/tests/ -n $((`nproc --all`/5+1)) &&
pytest {project}/faiss/tests/torch_test_contrib.py -n $((`nproc --all`/5+1)) &&
# GPU Test
pytest {project}/faiss/tests/common_faiss_tests.py {project}/faiss/faiss/gpu/test/ -n 4 &&
pytest {project}/faiss/faiss/gpu/test/torch_test_contrib_gpu.py
"""
        if variant == "gpu-cu11":
            test_requires = ["--extra-index-url", "https://download.pytorch.org/whl/cu118"]
        elif variant == "gpu-cu12":
            test_requires = ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
    pyproject["project"]["name"] = f"faiss-{args.variant}"
    pyproject["project"]["dependencies"] += dependencies
    pyproject["project"]["optional-dependencies"] = optional_dependencies
    pyproject["project"]["classifiers"] += classifiers
    repair_option = " ".join(
        [i for v in config["python"]["preload-library"] for i in ["--exclude", v["library"]]]
    )
    pyproject["tool"]["cibuildwheel"]["linux"]["before-all"] = (
        f"{' '.join([f'{k}="{v}"' for k, v in build_envs.items()])} script/build.sh"
    )
    pyproject["tool"]["cibuildwheel"]["linux"]["environment-pass"] += enviromnet_pass
    pyproject["tool"]["cibuildwheel"]["linux"]["environment"] |= envs
    pyproject["tool"]["cibuildwheel"]["linux"]["test-requires"] += test_requires
    pyproject["tool"]["cibuildwheel"]["linux"] |= {
        "repair-wheel-command": f"auditwheel repair -w {{dest_dir}} {{wheel}} {repair_option}",
        "test-command": test_command,
    }

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
