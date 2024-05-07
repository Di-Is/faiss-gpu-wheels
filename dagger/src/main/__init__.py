"""Serve dagger CI pipeline
Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import json
import logging
from os.path import dirname

import dagger
from dagger import dag, function, object_type
from dagger.log import configure_logging
from anyio.from_thread import start_blocking_portal

import main.test
from main import gpu_builder, cpu_builder
from main.builder import AbsWheelBuilder

configure_logging(logging.DEBUG)

DAGGER_ROOT = dirname(dirname(dirname(__file__)))


@object_type
class FaissWheels:
    @function
    async def build_cpu_container(
        self, host_directory: dagger.Directory
    ) -> dagger.Container:
        """Build faiss-cpu build image

        Args:
            host_directory: host environment directory

        Returns:
            container for faiss-cpu building
        """
        cfg = self._load_cpu_config()

        container = await cpu_builder.ImageBuilder(
            dag,
            host_directory,
            cfg["build"],
            cfg["auditwheel"],
        ).build()

        await container.sync()
        return container

    @function
    async def build_cpu_wheels(
        self,
        host_directory: dagger.Directory,
    ) -> dagger.Directory:
        """Build faiss-cpu wheels

        Args:
            host_directory: host environment directory

        Returns:
            directory included faiss-cpu wheels
        """
        cfg = self._load_cpu_config()

        container = await self.build_cpu_container(host_directory)

        # make wheel
        wheel_builder = cpu_builder.WheelBuilder(
            container,
            host_directory,
            cfg["build"],
            cfg["auditwheel"],
            cfg["python"],
        )

        wheel_files = await self._build_wheels(
            wheel_builder,
            cfg["python"]["support_versions"],
            cfg["build"]["parallel_wheel_build"],
        )
        return dag.directory().with_files(".", wheel_files)

    @function
    async def build_gpu_container(
        self, cuda_major: int, host_directory: dagger.Directory
    ) -> dagger.Container:
        """Build faiss-gpu build image

        Args:
            cuda_major: build target CUDA major version
            host_directory: host environment directory

        Returns:
            container for faiss-gpu building
        """
        cfg = self._load_gpu_config(cuda_major)

        container = await gpu_builder.ImageBuilder(
            dag,
            host_directory,
            cfg["build"],
            cfg["auditwheel"],
            cfg["cuda"],
        ).build()

        await container.sync()
        return container

    @function
    async def build_gpu_wheels(
        self,
        cuda_major: int,
        host_directory: dagger.Directory,
    ) -> dagger.Directory:
        """Build faiss-gpu wheels

        Args:
            cuda_major: build target CUDA major version
            host_directory: host environment directory

        Returns:
            directory included faiss-gpu wheels
        """
        # build image for faiss-gpu wheel building
        container = await self.build_gpu_container(cuda_major, host_directory)

        # build wheel
        cfg = self._load_gpu_config(cuda_major)
        wheel_builder = gpu_builder.WheelBuilder(
            container,
            host_directory,
            cfg["build"],
            cfg["auditwheel"],
            cfg["python"],
            cfg["cuda"],
        )
        wheel_files = await self._build_wheels(
            wheel_builder,
            cfg["python"]["support_versions"],
            cfg["build"]["parallel_wheel_build"],
        )

        return dag.directory().with_files(".", wheel_files)

    @function
    async def test_gpu_wheels(self, cuda_major: int, host_directory: dagger.Directory):
        """test faiss-gpu wheels

        Args:
            cuda_major: build target CUDA major version
            host_directory: host environment directory
        """
        cfg = self._load_gpu_config(cuda_major)
        cfg = _expand_test_config(cfg)

        faiss_ver = await host_directory.file("version.txt").contents()
        faiss_ver = faiss_ver.replace("\n", "")

        whlname_maker = gpu_builder.WheelName(
            faiss_ver, cfg["auditwheel"]["policy"], cfg["cuda"]["major_version"]
        )
        # uv package manager cache
        uv_cache = dag.cache_volume("uv-cache")

        for test_name, test_cfg in cfg["test"].items():
            print(f"Test case {test_name} Start")

            container = (
                dag.container().from_(test_cfg["image"]).experimental_with_gpu(["0"])
            )
            container = await main.test.install_uv(container)
            await container.sync()

            whl_name = whlname_maker.make_repaired_wheelname(
                test_cfg["target_python_version"]
            )

            container = (
                container.with_directory("/project", host_directory)
                .with_workdir("project")
                .with_env_variable("UV_CACHE_DIR", "/tmp/uv_cache")
                .with_mounted_cache("/tmp/uv_cache", uv_cache)
                .with_exec(
                    [
                        "uv",
                        "pip",
                        "install",
                        f"wheelhouse/gpu/cuda{cuda_major}/{whl_name}[fix_cuda]",
                    ]
                    + test_cfg["requires"]
                )
                .with_env_variable("OMP_NUM_THREADS", "1")
            )
            await container.sync()

            for test_name in test_cfg["cases"]:
                func = getattr(main.test, test_name)
                print(await func(container))

            print(f"Test case {test_name} End")

    @function
    async def test_cpu_wheels(self, host_directory: dagger.Directory):
        """test faiss-cpu wheels

        Args:
            host_directory: host environment directory
        """
        cfg = self._load_cpu_config()
        cfg = _expand_test_config(cfg)

        faiss_ver = await host_directory.file("version.txt").contents()
        faiss_ver = faiss_ver.replace("\n", "")

        whlname_maker = cpu_builder.WheelName(faiss_ver, cfg["auditwheel"]["policy"])
        # uv package manager cache
        uv_cache = dag.cache_volume("uv-cache")

        for test_name, test_cfg in cfg["test"].items():
            print(f"Test case {test_name} Start")

            container = dag.container().from_(test_cfg["image"])
            container = await main.test.install_uv(container)
            await container.sync()

            whl_name = whlname_maker.make_repaired_wheelname(
                test_cfg["target_python_version"]
            )

            container = (
                container.with_directory("/project", host_directory)
                .with_workdir("project")
                .with_env_variable("UV_CACHE_DIR", "/tmp/uv_cache")
                .with_mounted_cache("/tmp/uv_cache", uv_cache)
                .with_exec(
                    [
                        "uv",
                        "pip",
                        "install",
                        f"wheelhouse/cpu/{whl_name}",
                    ]
                    + test_cfg["requires"]
                )
                .with_env_variable("OMP_NUM_THREADS", "1")
            )
            await container.sync()

            for test_name in test_cfg["cases"]:
                func = getattr(main.test, test_name)
                print(await func(container))

            print(f"Test case {test_name} End")

    @classmethod
    async def _build_wheels(
        cls,
        wheel_builder: AbsWheelBuilder,
        python_versions: list[str],
        parallel: bool = False,
    ) -> list[dagger.File]:
        if parallel:
            with start_blocking_portal() as tg:
                results = [
                    tg.start_task_soon(
                        cls._build_wheel_specific_python, wheel_builder, py_ver
                    )
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
        """build faiss gpu wheel
        Args:
            version: build target python version ({major}.{minor})

        Returns:
            built wheel file
        """
        file = await wheel_builder.build(python_version)
        return file

    @staticmethod
    def _load_gpu_config(cuda_major: int) -> dict:
        """load gpu config

        Returns:
            gpu config
        """
        if cuda_major not in [11, 12]:
            raise ValueError(f"Incompatible cuda version. {cuda_major}")

        with open(f"{DAGGER_ROOT}/config/gpu.cuda{cuda_major}.json", "r") as f:
            cfg = json.load(f)
        return cfg

    @staticmethod
    def _load_cpu_config() -> str:
        """load cpu config

        Returns:
            cpu config
        """
        with open(f"{DAGGER_ROOT}/config/cpu.json", "r") as f:
            cfg = json.load(f)
        return cfg


def _expand_test_config(cfg: dict) -> dict:
    """recognize special characters and expand test settings

    Args:
        cfg: config dict

    Returns:
        expaned condig dict
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
