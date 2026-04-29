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
from functools import cache
from pathlib import Path
from typing import Literal

import click
import requests
import tomli_w
import tomllib
from pydanclick import from_pydantic
from pydantic import BaseModel

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

_FAISS_TESTS_CMD = (
    "pytest {project}/faiss/tests/ -n $((`nproc --all`/5+1))"
    ' ${FAISS_TESTS_EXTRA_OPTS:+-k "$FAISS_TESTS_EXTRA_OPTS"}'
)

_CPU_TEST_COMMAND = f"""
# CPU Test
{_FAISS_TESTS_CMD} &&
pytest {{project}}/faiss/tests/torch_test_contrib.py -n $((`nproc --all`/5+1))
"""

_GPU_TEST_COMMAND = f"""
# CPU Test
{_FAISS_TESTS_CMD} &&
pytest {{project}}/faiss/tests/torch_test_contrib.py -n $((`nproc --all`/5+1)) &&
# GPU Test
pytest {{project}}/faiss/tests/common_faiss_tests.py {{project}}/faiss/faiss/gpu/test/ &&
pytest {{project}}/faiss/faiss/gpu/test/torch_test_contrib_gpu.py
"""


@cache
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


def _build_cuvs_envs() -> dict[str, str]:
    return {
        "FAISS_ENABLE_CUVS": "ON",
        "CMAKE_PREFIX_PATH": ";".join(
            [
                f"{RAPIDS_WHEEL_DIR}/libcuvs",
                f"{RAPIDS_WHEEL_DIR}/libraft",
                f"{RAPIDS_WHEEL_DIR}/librmm",
                f"{RAPIDS_WHEEL_DIR}/rapids_logger",
            ]
        ),
        "FAISS_CUVS_DIR": f"{RAPIDS_WHEEL_DIR}/libcuvs/lib64/cmake/cuvs",
        "FAISS_RAFT_DIR": f"{RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/raft",
        "FAISS_RMM_DIR": f"{RAPIDS_WHEEL_DIR}/librmm/lib64/cmake/rmm",
        "FAISS_HNSWLIB_DIR": f"{RAPIDS_WHEEL_DIR}/libcuvs/lib64/cmake/hnswlib",
        "FAISS_CUCO_DIR": f"{RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/cuco",
        "FAISS_NVIDIA_CUTLASS_DIR": f"{RAPIDS_WHEEL_DIR}/libraft/lib64/cmake/NvidiaCutlass",
        "FAISS_NVTX3_DIR": f"{RAPIDS_WHEEL_DIR}/librmm/lib64/cmake/nvtx3",
        "FAISS_RAPIDS_LOGGER_DIR": (f"{RAPIDS_WHEEL_DIR}/rapids_logger/lib64/cmake/rapids_logger"),
        "FAISS_C_COMPILER": f"{GCC_TOOLSET_13_BIN}/gcc",
        "FAISS_CXX_COMPILER": f"{GCC_TOOLSET_13_BIN}/g++",
        "FAISS_CUDA_HOST_COMPILER": f"{GCC_TOOLSET_13_BIN}/g++",
    }


def _build_gpu_variant_config(config: dict, variant: str) -> dict:
    major, _, _ = config["cuda"]["version"].split(".")
    dependencies = [
        get_cuda_package_spec(config["cuda"]["version"], major, "cuda_cudart", ">="),
        get_cuda_package_spec(config["cuda"]["version"], major, "libcublas", ">="),
    ]
    if variant == "gpu-cuvs":
        dependencies.append(f"libcuvs-cu{major}=={config['cuvs']['version']}")
    fix_cuda_components = list(
        dict.fromkeys(
            ["cuda_cudart", "libcublas", *config["python"].get("fix-cuda-components", [])]
        )
    )
    optional_dependencies: dict[str, list] = {
        "fix-cuda": [
            get_cuda_package_spec(config["cuda"]["version"], major, component, "==")
            for component in fix_cuda_components
        ]
    }
    build_envs: dict[str, str] = {
        "CUDA_VERSION": config["cuda"]["version"],
        "CUDA_ARCHITECTURES": config["cuda"]["target-archs"],
    }
    envs: dict[str, str] = {"FAISS_ENABLE_GPU": "ON"}
    if variant == "gpu-cuvs":
        build_envs["CUVS_VERSION"] = config["cuvs"]["version"]
        build_envs["NVIDIA_EXTRA_INDEX_URL"] = config["python"]["extra-index-url"][0]
        build_envs["CMAKE_VERSION"] = config["build"]["cmake-version"]
        build_envs["RAPIDS_WHEEL_DIR"] = RAPIDS_WHEEL_DIR
        envs["PIP_EXTRA_INDEX_URL"] = config["python"]["extra-index-url"][0]
        envs["UV_EXTRA_INDEX_URL"] = config["python"]["extra-index-url"][0]
        envs.update(_build_cuvs_envs())
    return {
        "dependencies": dependencies,
        "optional_dependencies": optional_dependencies,
        "classifiers": [
            "Environment :: GPU :: NVIDIA CUDA",
            f"Environment :: GPU :: NVIDIA CUDA :: {major}",
        ],
        "build_envs": build_envs,
        "envs": envs,
        "environment_pass": ["CUDA_ARCHITECTURES", "FAISS_TESTS_EXTRA_OPTS"],
        "test_extras": ["fix-cuda"],
        "test_command": _GPU_TEST_COMMAND,
    }


def _build_repair_command(config: dict) -> str:
    repair_excludes = list(
        dict.fromkeys(
            [v["library"] for v in config["python"]["preload-library"] if "library" in v]
            + config["python"].get("auditwheel-exclude", [])
        )
    )
    repair_option = " ".join([i for library in repair_excludes for i in ["--exclude", library]])
    repair_command = f"auditwheel repair -w {{dest_dir}} {{wheel}} {repair_option}".strip()
    repair_command_prefix = config["python"].get("repair-wheel-command-prefix", "")
    if repair_command_prefix:
        repair_command = f"{repair_command_prefix} {repair_command}"
    return repair_command


def _build_uv_config(config: dict) -> dict[str, object]:
    uv_indexes: list[dict[str, object]] = [
        {"name": "torch-index", "url": config["test"]["index-url"], "explicit": True}
    ]
    explicit_extra_indexes = set(config["python"].get("explicit-extra-index-url", []))
    for index_no, extra_index_url in enumerate(
        config["python"].get("extra-index-url", []), start=1
    ):
        uv_index: dict[str, object] = {
            "name": f"extra-index-{index_no}",
            "url": extra_index_url,
        }
        if index_no in explicit_extra_indexes:
            uv_index["explicit"] = True
        uv_indexes.append(uv_index)
    uv_sources: dict[str, object] = {"torch": {"index": "torch-index"}}
    for package_name, index_name in config["python"].get("source-indexes", {}).items():
        uv_sources[package_name] = {"index": index_name}
    uv_config: dict[str, object] = {"index": uv_indexes, "sources": uv_sources}
    environments = config["python"].get("environments", [])
    if environments:
        uv_config["environments"] = environments
    required_environments = config["python"].get("required-environments", [])
    if required_environments:
        uv_config["required-environments"] = required_environments
    return uv_config


class Args(BaseModel):
    """Script argument."""

    variant: Literal["cpu", "gpu-cu11", "gpu-cu12", "gpu-cuvs"]


@click.command()
@from_pydantic(Args)
def cli(args: Args) -> None:
    """CLI function."""
    variant = args.variant
    variant_path = Path(__file__).parent.parent / "variant" / f"{variant}"
    with (variant_path / "config.toml").open("rb") as f:
        config = tomllib.load(f)
    with (Path(__file__).parent / "template" / "pyproject.toml.tpl").open("rb") as f:
        pyproject = tomllib.load(f)
    preload_config_path = variant_path / "_preload_library.json"
    with preload_config_path.open("w") as f:
        json.dump({"preload-library": config["python"]["preload-library"]}, f, indent=4)

    if variant == "cpu":
        vc: dict = {
            "dependencies": [],
            "optional_dependencies": {},
            "classifiers": [],
            "build_envs": {},
            "envs": {},
            "environment_pass": ["FAISS_TESTS_EXTRA_OPTS"],
            "test_extras": [],
            "test_command": _CPU_TEST_COMMAND,
        }
    else:
        vc = _build_gpu_variant_config(config, variant)

    pyproject["project"]["name"] = f"faiss-{args.variant}"
    pyproject["project"]["dependencies"] += vc["dependencies"]
    pyproject["project"]["optional-dependencies"] = vc["optional_dependencies"]
    pyproject["project"]["classifiers"] += vc["classifiers"]
    env_vars = " ".join([f'{k}="{v}"' for k, v in vc["build_envs"].items()])
    pyproject["tool"]["cibuildwheel"]["linux"]["before-all"] = f"{env_vars} script/build.sh"
    pyproject["tool"]["cibuildwheel"]["linux"]["environment-pass"] += vc["environment_pass"]
    pyproject["tool"]["cibuildwheel"]["linux"]["environment"] |= vc["envs"]
    pyproject["tool"]["cibuildwheel"]["linux"]["before-test"] = (
        f"uv sync --project variant/{variant} --no-install-project --active"
    )
    pyproject["tool"]["cibuildwheel"]["linux"]["test-extras"] = vc["test_extras"]
    pyproject["tool"]["cibuildwheel"]["linux"] |= {
        "repair-wheel-command": _build_repair_command(config),
        "test-command": vc["test_command"],
    }
    pyproject["tool"]["cibuildwheel"] |= config.get("cibuildwheel", {})
    pyproject["dependency-groups"]["dev"] = config["test"]["dependencies"]
    pyproject["tool"]["uv"] = _build_uv_config(config)
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
