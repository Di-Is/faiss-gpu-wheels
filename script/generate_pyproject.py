#!/usr/bin/env -S uv run --script -q
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

GPU_VARIANTS = {"gpu-cu11", "gpu-cu12", "gpu-cuvs"}
RAPIDS_WHEEL_DIR = "/usr/local/rapids-wheel-deps"
GCC_TOOLSET_13_BIN = "/opt/rh/gcc-toolset-13/root/usr/bin"
CUDA_COMPONENT_PACKAGES = {
    "cuda_cudart": "nvidia-cuda-runtime-cu{major}",
    "libcublas": "nvidia-cublas-cu{major}",
    "libcurand": "nvidia-curand-cu{major}",
    "libcusolver": "nvidia-cusolver-cu{major}",
    "libcusparse": "nvidia-cusparse-cu{major}",
    "libnvjitlink": "nvidia-nvjitlink-cu{major}",
}


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
        url = f"https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/{json_file}"
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


def get_cuda_package_spec(version: str, major: str, component: str, operator: str) -> str:
    """Return a CUDA Python package spec for the requested toolkit component."""
    package = CUDA_COMPONENT_PACKAGES[component].format(major=major)
    component_version = get_cuda_version(version, component)
    return f"{package}{operator}{component_version}"


class Args(BaseModel):
    """Script argument."""

    variant: Literal["cpu", "gpu-cu11", "gpu-cu12", "gpu-cuvs"]


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
        enviromnet_pass = []
        test_extras = []
        test_command = """
# CPU Test
pytest {project}/faiss/tests/ -n $((`nproc --all`/5+1)) &&
pytest {project}/faiss/tests/torch_test_contrib.py -n $((`nproc --all`/5+1))
"""
    elif variant in GPU_VARIANTS:
        major, _, _ = config["cuda"]["version"].split(".")
        dependencies = [
            get_cuda_package_spec(config["cuda"]["version"], major, "cuda_cudart", ">="),
            get_cuda_package_spec(config["cuda"]["version"], major, "libcublas", ">="),
        ]
        if variant == "gpu-cuvs":
            dependencies.append(f"libcuvs-cu{major}=={config['cuvs']['version']}")
        fix_cuda_components = ["cuda_cudart", "libcublas"]
        fix_cuda_components += config["python"].get("fix-cuda-components", [])
        optional_dependencies = {"fix-cuda": []}
        seen_specs: set[str] = set()
        for component in fix_cuda_components:
            spec = get_cuda_package_spec(config["cuda"]["version"], major, component, "==")
            if spec in seen_specs:
                continue
            seen_specs.add(spec)
            optional_dependencies["fix-cuda"].append(spec)
        test_extras = ["fix-cuda"]
        classifiers = [
            "Environment :: GPU :: NVIDIA CUDA",
            f"Environment :: GPU :: NVIDIA CUDA :: {major}",
        ]
        build_envs = {"CUDA_VERSION": config["cuda"]["version"]}
        envs = {"FAISS_ENABLE_GPU": "ON"}
        if variant == "gpu-cuvs":
            build_envs["CUVS_VERSION"] = config["cuvs"]["version"]
            build_envs["NVIDIA_EXTRA_INDEX_URL"] = config["python"]["extra-index-url"][0]
            build_envs["CMAKE_VERSION"] = config["build"]["cmake-version"]
            build_envs["RAPIDS_WHEEL_DIR"] = RAPIDS_WHEEL_DIR
            envs["FAISS_ENABLE_CUVS"] = "ON"
            envs["PIP_EXTRA_INDEX_URL"] = config["python"]["extra-index-url"][0]
            envs["UV_EXTRA_INDEX_URL"] = config["python"]["extra-index-url"][0]
            envs["CMAKE_PREFIX_PATH"] = ";".join(
                [
                    f"{RAPIDS_WHEEL_DIR}/libcuvs",
                    f"{RAPIDS_WHEEL_DIR}/libraft",
                    f"{RAPIDS_WHEEL_DIR}/librmm",
                    f"{RAPIDS_WHEEL_DIR}/rapids_logger",
                ]
            )
            envs["FAISS_CUVS_DIR"] = f"{RAPIDS_WHEEL_DIR}/libcuvs/lib64/cmake/cuvs"
            envs["FAISS_RAFT_DIR"] = f"{RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/raft"
            envs["FAISS_RMM_DIR"] = f"{RAPIDS_WHEEL_DIR}/librmm/lib64/cmake/rmm"
            envs["FAISS_HNSWLIB_DIR"] = f"{RAPIDS_WHEEL_DIR}/libcuvs/lib64/cmake/hnswlib"
            envs["FAISS_CUCO_DIR"] = f"{RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/cuco"
            envs["FAISS_NVIDIA_CUTLASS_DIR"] = (
                f"{RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/NvidiaCutlass"
            )
            envs["FAISS_NVTX3_DIR"] = f"{RAPIDS_WHEEL_DIR}/librmm/lib64/cmake/nvtx3"
            envs["FAISS_RAPIDS_LOGGER_DIR"] = (
                f"{RAPIDS_WHEEL_DIR}/rapids_logger/lib64/cmake/rapids_logger"
            )
            envs["FAISS_C_COMPILER"] = f"{GCC_TOOLSET_13_BIN}/gcc"
            envs["FAISS_CXX_COMPILER"] = f"{GCC_TOOLSET_13_BIN}/g++"
            envs["FAISS_CUDA_HOST_COMPILER"] = f"{GCC_TOOLSET_13_BIN}/g++"
        enviromnet_pass = ["CUDA_ARCHITECTURES"]
        test_command = """
# CPU Test
pytest {project}/faiss/tests/ -n $((`nproc --all`/5+1)) &&
pytest {project}/faiss/tests/torch_test_contrib.py -n $((`nproc --all`/5+1)) &&
# GPU Test
pytest {project}/faiss/tests/common_faiss_tests.py {project}/faiss/faiss/gpu/test/ &&
pytest {project}/faiss/faiss/gpu/test/torch_test_contrib_gpu.py
"""
    pyproject["project"]["name"] = f"faiss-{args.variant}"
    pyproject["project"]["dependencies"] += dependencies
    pyproject["project"]["optional-dependencies"] = optional_dependencies
    pyproject["project"]["classifiers"] += classifiers
    repair_excludes = [
        v["library"] for v in config["python"]["preload-library"] if "library" in v
    ] + config["python"].get("auditwheel-exclude", [])
    repair_option = " ".join([i for library in repair_excludes for i in ["--exclude", library]])
    env_vars = " ".join([f'{k}="{v}"' for k, v in build_envs.items()])
    pyproject["tool"]["cibuildwheel"]["linux"]["before-all"] = f"{env_vars} script/build.sh"
    pyproject["tool"]["cibuildwheel"]["linux"]["environment-pass"] += enviromnet_pass
    pyproject["tool"]["cibuildwheel"]["linux"]["environment"] |= envs
    pyproject["tool"]["cibuildwheel"]["linux"]["before-test"] = (
        f"uv sync --project variant/{variant} --no-install-project --active"
    )
    pyproject["tool"]["cibuildwheel"]["linux"]["test-extras"] = test_extras
    pyproject["tool"]["cibuildwheel"]["linux"] |= {
        "repair-wheel-command": f"auditwheel repair -w {{dest_dir}} {{wheel}} {repair_option}",
        "test-command": test_command,
    }
    pyproject["tool"]["cibuildwheel"] |= config.get("cibuildwheel", {})
    pyproject["dependency-groups"]["dev"] = config["test"]["dependencies"]
    uv_indexes: list[dict[str, object]] = [
        {"name": "torch-index", "url": config["test"]["index-url"], "explicit": True}
    ]
    for index_no, extra_index_url in enumerate(config["python"].get("extra-index-url", []), start=1):
        uv_indexes.append({"name": f"extra-index-{index_no}", "url": extra_index_url})
    pyproject["tool"]["uv"] = {"index": uv_indexes, "sources": {"torch": {"index": "torch-index"}}}
    pyproject_path = variant_path / "pyproject.toml"
    text = """# Copyright (c) 2024 Di-Is
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
# This file is created by the `generate_pyproject.py`."""
    pyproject_path.write_text(f"{text}\n" + tomli_w.dumps(pyproject))

    _ = subprocess.run(  # noqa: S603
        ["taplo", "format", str(pyproject_path)],  # noqa: S607
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    cli()
