"""Serve dagger CI pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from anyio.from_thread import start_blocking_portal

import dagger
import faiss_wheels.test
from dagger import dag, function, object_type
from faiss_wheels import cpu_builder, gpu_builder

from ._static_analysis import static_analysis
from ._util import UV_CACHE, UV_VERSION, install_uv

if TYPE_CHECKING:
    from faiss_wheels.builder import AbsWheelBuilder


DAGGER_ROOT = Path(__file__).parent.parent


@object_type
class FaissWheels:
    """dagger ci piplline."""

    ruff_version = "0.7.0"

    @function
    async def static_analysis(self, source: dagger.Directory) -> None:
        """Execute static analysis.

        Args:
            source: source directory
        """
        await static_analysis(source)

    @function
    async def faiss_cpu_ci(self, host_directory: dagger.Directory) -> dagger.Directory:
        """faiss-cpu ci pipeline.

        Args:
            host_directory: host environment directory

        Returns:
            directory included faiss-cpu wheels
        """
        wheel_dir = await self.build_cpu_wheels(host_directory)
        await self.test_cpu_wheels(host_directory, wheel_dir)
        return wheel_dir

    @function
    async def faiss_gpu_ci(
        self, host_directory: dagger.Directory, cuda_major_versions: list[int]
    ) -> dagger.Directory:
        """faiss-gpu ci pipeline.

        Args:
            host_directory: host environment directory
            cuda_major_versions: build target cuda major versions

        Returns:
            directory included faiss-gpu wheels
        """
        result: dict[int, dagger.Directory] = {}
        for cu_ver in cuda_major_versions:
            wheel_dir = await self.build_gpu_wheels(host_directory, cu_ver)
            await self.test_gpu_wheels(host_directory, wheel_dir, cu_ver)
            result[cu_ver] = wheel_dir

        # combine wheels directory
        wheel_dir = dag.directory()
        for cu_ver, whl_dir in result.items():
            wheel_dir = wheel_dir.with_directory(f"gpu/cuda{cu_ver}", whl_dir)
        return wheel_dir

    @function
    async def build_cpu_container(self, host_directory: dagger.Directory) -> dagger.Container:
        """Build faiss-cpu build image.

        Args:
            host_directory: host environment directory

        Returns:
            container for faiss-cpu building
        """
        cfg = self._load_cpu_config()

        container = await cpu_builder.ImageBuilder(
            host_directory, cfg["build"], cfg["auditwheel"]
        ).build()

        await container.sync()
        return container

    @function
    async def build_cpu_wheels(
        self,
        host_directory: dagger.Directory,
    ) -> dagger.Directory:
        """Build faiss-cpu wheels.

        Args:
            host_directory: host environment directory

        Returns:
            directory included faiss-cpu wheels
        """
        cfg = self._load_cpu_config()

        container = (
            install_uv(await self.build_cpu_container(host_directory), UV_VERSION)
        ).with_env_variable("UV_PYTHON_PREFERENCE", "only-system")

        # make wheel
        wheel_builder = cpu_builder.WheelBuilder(
            container, host_directory, cfg["build"], cfg["auditwheel"], cfg["python"]
        )

        wheel_files = await self._build_wheels(
            wheel_builder,
            cfg["python"]["support_versions"],
            parallel=cfg["build"]["parallel_wheel_build"],
        )
        return dag.directory().with_files(".", wheel_files)

    @function
    async def build_gpu_container(
        self,
        host_directory: dagger.Directory,
        cuda_major_version: int,
    ) -> dagger.Container:
        """Build faiss-gpu build image.

        Args:
            host_directory: host environment directory
            cuda_major_version: build target CUDA major version

        Returns:
            container for faiss-gpu building
        """
        cfg = self._load_gpu_config(cuda_major_version)

        container = await gpu_builder.ImageBuilder(
            host_directory, cfg["build"], cfg["auditwheel"], cfg["cuda"]
        ).build()

        await container.sync()
        return container

    @function
    async def build_gpu_wheels(
        self,
        host_directory: dagger.Directory,
        cuda_major_version: int,
    ) -> dagger.Directory:
        """Build faiss-gpu wheels.

        Args:
            host_directory: host environment directory
            cuda_major_version: build target CUDA major version

        Returns:
            directory included faiss-gpu wheels
        """
        # build image for faiss-gpu wheel building
        container = install_uv(
            await self.build_gpu_container(host_directory, cuda_major_version),
            UV_VERSION,
        ).with_env_variable("UV_PYTHON_PREFERENCE", "only-system")

        # build wheel
        cfg = self._load_gpu_config(cuda_major_version)
        wheel_builder = gpu_builder.WheelBuilder(
            container, host_directory, cfg["build"], cfg["auditwheel"], cfg["python"], cfg["cuda"]
        )
        wheel_files = await self._build_wheels(
            wheel_builder,
            cfg["python"]["support_versions"],
            parallel=cfg["build"]["parallel_wheel_build"],
        )

        return dag.directory().with_files(".", wheel_files)

    @function
    async def test_gpu_wheels(
        self,
        host_directory: dagger.Directory,
        wheel_directory: dagger.Directory,
        cuda_major_version: int,
    ) -> None:
        """Test faiss-gpu wheels.

        Args:
            host_directory: host environment directory
            wheel_directory: directory included wheels
            cuda_major_version: build target CUDA major version
        """
        cfg = self._load_gpu_config(cuda_major_version)
        cfg = _expand_test_config(cfg)

        for test_cfg in cfg["test"].values():
            ctr = install_uv(
                dag.container().from_(test_cfg["image"]).experimental_with_gpu(["0"]),
                UV_VERSION,
            )
            py = "cp" + test_cfg["target_python_version"].replace(".", "")
            wheel_name = (await wheel_directory.glob(f"*{py}*.whl"))[0]
            wheel = wheel_directory.file(wheel_name)

            ctr = (
                ctr.with_directory("/project", host_directory)
                .with_workdir("project")
                .with_mounted_file(wheel_name, wheel)
                .with_env_variable("UV_CACHE_DIR", "/root/.cache/uv")
                .with_env_variable("UV_SYSTEM_PYTHON", "true")
                .with_env_variable("UV_HTTP_TIMEOUT", "10000000")
                .with_mounted_cache("/root/.cache/uv", UV_CACHE)
                .with_exec(
                    ["uv", "pip", "install", f"{wheel_name}[fix-cuda]"] + test_cfg["requires"]
                )
                .with_env_variable("OMP_NUM_THREADS", "1")
            )

            for case_name in test_cfg["cases"]:
                func = getattr(faiss_wheels.test, case_name)
                await func(ctr)

    @function
    async def test_cpu_wheels(
        self, host_directory: dagger.Directory, wheel_directory: dagger.Directory
    ) -> None:
        """Test faiss-cpu wheels.

        Args:
            host_directory: host environment directory
            wheel_directory: directory included wheels
        """
        cfg = self._load_cpu_config()
        cfg = _expand_test_config(cfg)

        for test_cfg in cfg["test"].values():
            ctr = install_uv(dag.container().from_(test_cfg["image"]), UV_VERSION)
            py = "cp" + test_cfg["target_python_version"].replace(".", "")
            wheel_name = (await wheel_directory.glob(f"*{py}*.whl"))[0]
            wheel = wheel_directory.file(wheel_name)

            ctr = (
                ctr.with_directory("/project", host_directory)
                .with_workdir("project")
                .with_mounted_file(wheel_name, wheel)
                .with_env_variable("UV_CACHE_DIR", "/root/.cache/uv")
                .with_env_variable("UV_SYSTEM_PYTHON", "true")
                .with_env_variable("UV_HTTP_TIMEOUT", "10000000")
                .with_mounted_cache("/root/.cache/uv", UV_CACHE)
                .with_exec(["uv", "pip", "install", wheel_name] + test_cfg["requires"])
                .with_env_variable("OMP_NUM_THREADS", "1")
            )

            for case_name in test_cfg["cases"]:
                func = getattr(faiss_wheels.test, case_name)
                await func(ctr)

    @classmethod
    async def _build_wheels(
        cls, wheel_builder: AbsWheelBuilder, python_versions: list[str], *, parallel: bool = False
    ) -> list[dagger.File]:
        """Build wheel.

        Args:
            wheel_builder: concrete class of AbsWheelBuilder
            python_versions: build target python versions
            parallel: flag of parallel wheel building. Defaults to False.

        Returns:
            _description_
        """
        if parallel:
            with start_blocking_portal() as tg:
                results = [
                    tg.start_task_soon(cls._build_wheel_specific_python, wheel_builder, py_ver)
                    for py_ver in python_versions
                ]
                wheel_files = [f.result() for f in results]
        else:
            wheel_files = [
                await cls._build_wheel_specific_python(wheel_builder, py_ver)
                for py_ver in python_versions
            ]

        return wheel_files

    @staticmethod
    async def _build_wheel_specific_python(
        wheel_builder: AbsWheelBuilder, python_version: str
    ) -> dagger.File:
        """Build faiss gpu wheel.

        Args:
            wheel_builder: a
            python_version: a

        Returns:
            built wheel file
        """
        return await wheel_builder.build(python_version)

    @staticmethod
    def _load_gpu_config(cuda_major_version: int) -> dict:
        """Load gpu config.

        Returns:
            gpu config
        """
        if cuda_major_version not in [11, 12]:
            msg = f"Incompatible cuda version. {cuda_major_version}"
            raise ValueError(msg)

        path = Path(DAGGER_ROOT) / "config" / f"gpu.cuda{cuda_major_version}.json"
        with path.open("r") as f:
            return json.load(f)

    @staticmethod
    def _load_cpu_config() -> str:
        """Load cpu config.

        Returns:
            cpu config
        """
        path = Path(DAGGER_ROOT) / "config" / "cpu.json"
        with path.open("r") as f:
            return json.load(f)


def _expand_test_config(cfg: dict) -> dict:
    """Recognize special characters and expand test settings.

    Args:
        cfg: config dict

    Returns:
        expand config dict
    """
    cfg_test = {}
    for name, val in cfg["test"].items():
        if val["target_python_version"] == "ALL":
            for ver in cfg["python"]["support_versions"]:
                v = val.copy()
                v["target_python_version"] = ver
                v["image"] = v["image"].replace("{target_python_version}", ver)
                cfg_test[f"{name}_{ver}"] = v
        else:
            cfg_test[name] = val
    cfg["test"] = cfg_test
    return cfg
