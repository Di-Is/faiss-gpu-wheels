"""Serve faiss-gpu building function using dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import logging
from pathlib import Path

from dagger import BuildArg, Container, Directory, File
from dagger.client.gen import Client
from dagger.log import configure_logging

from .builder import AbsWheelBuilder
from .type import AuditWheelConfig, BuildConfig, CUDAConfig, PythonConfig

configure_logging(logging.DEBUG)


class ImageBuilder:
    """image builder for faiss-gpu."""

    def __init__(
        self,
        client: Client,
        host_directory: Directory,
        build_config: BuildConfig,
        auditwheel_config: AuditWheelConfig,
        cuda_config: CUDAConfig,
    ) -> None:
        """constructor.

        Args:
            client: dagger client
            host_directory: repository root directory
            build_config: build config
            auditwheel_config: auditwheel config
            cuda_config: cuda config
        """
        self._client = client
        self._host_directory = host_directory
        self._build_config = build_config
        self._auditwheel_config = auditwheel_config
        self._cuda_config = cuda_config

    async def build(self) -> Container:
        """Build image using faiss-gpu wheel building.

        Returns:
            container for faiss-gpu wheel building
        """
        # build wheel builder image
        builder = self._host_directory.docker_build(
            dockerfile="./docker/Dockerfile.gpu",
            build_args=[
                BuildArg("CUDA_MAJOR_VERSION", self._cuda_config["major_version"]),
                BuildArg("CUDA_MINOR_VERSION", self._cuda_config["minor_version"]),
                BuildArg("CUDA_RUNTIME_VERSION", self._cuda_config["runtime_version"]),
                BuildArg("CUDA_CUBLAS_VERSION", self._cuda_config["cublas_version"]),
                BuildArg("CUDA_ARCHITECTURES", self._cuda_config["architectures"]),
                BuildArg("BUILD_NJOB", self._build_config["njob"]),
                BuildArg("FAISS_OPT_LEVEL", self._build_config["instruction_set"]),
                BuildArg(
                    "FAISS_ENABLE_GPU",
                    "ON" if self._build_config["enable_gpu"] else "OFF",
                ),
                BuildArg(
                    "FAISS_ENABLE_RAFT",
                    "ON" if self._build_config["enable_raft"] else "OFF",
                ),
                BuildArg("AUDITWHEEL_POLICY", self._auditwheel_config["policy"]),
            ],
        )
        await builder.sync()
        return builder


class WheelBuilder(AbsWheelBuilder):
    """faiss-gpu wheel builder."""

    _raw_dir: str = "/output/raw_wheels"
    _repaired_dir: str = "/output/repaired_wheels"

    def __init__(  # noqa: PLR0913
        self,
        builder: Container,
        host_directory: Directory,
        build_config: BuildConfig,
        auditwheel_config: AuditWheelConfig,
        python_config: PythonConfig,
        cuda_config: CUDAConfig,
    ) -> None:
        """constructor.

        Args:
            builder: container for faiss-gpu wheel building
            host_directory: repository root directory
            build_config: build config
            auditwheel_config: auditwheel config
            python_config: python config
            cuda_config: cuda config
        """
        self._builder = builder
        self._host_directory = host_directory
        self._build_config = build_config
        self._auditwheel_config = auditwheel_config
        self._python_config = python_config
        self._cuda_config = cuda_config

    async def build(self, python_target_version: str) -> File:
        """Build faiss-gpu wheel.

        Args:
            python_target_version: build target python version

        Returns:
            faiss-gpu wheel file
        """
        builder = (
            self._builder.with_directory("/project", self._host_directory)
            .with_workdir("/project")
            .with_env_variable("CUDA_MAJOR_VERSION", self._cuda_config["major_version"])
            .with_env_variable("CUDA_MINOR_VERSION", self._cuda_config["minor_version"])
            .with_env_variable(
                "CUDA_RUNTIME_VERSION", self._cuda_config["runtime_version"]
            )
            .with_env_variable(
                "CUDA_CUBLAS_VERSION", self._cuda_config["cublas_version"]
            )
            .with_env_variable(
                "CUDA_DYNAMIC_LINK", "ON" if self._cuda_config["dynamic_link"] else ""
            )
            .with_env_variable(
                "PYTHON_SUPPORT_VERSIONS",
                ";".join(self._python_config["support_versions"]),
            )
            .with_env_variable("FAISS_OPT_LEVEL", self._build_config["instruction_set"])
            .with_env_variable(
                "FAISS_ENABLE_GPU", "ON" if self._build_config["enable_gpu"] else ""
            )
            .with_env_variable(
                "FAISS_ENABLE_RAFT", "ON" if self._build_config["enable_raft"] else ""
            )
            .with_exec(["mkdir", "-p", self._raw_dir, self._repaired_dir])
        )

        builder = builder.with_exec(
            [
                f"python{python_target_version}",
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "-w",
                self._raw_dir,
            ]
        )

        # get faiss version
        faiss_ver = await self._host_directory.file("version.txt").contents()
        faiss_ver = faiss_ver.replace("\n", "")

        whlname_maker = WheelName(
            faiss_ver,
            self._auditwheel_config["policy"],
            self._cuda_config["major_version"],
        )

        builder = builder.with_exec(
            [
                "auditwheel",
                "repair",
                "-w",
                self._repaired_dir,
                str(
                    Path(self._raw_dir)
                    / whlname_maker.make_raw_wheelname(python_target_version)
                ),
            ]
            + self._auditwheel_config["repair_option"]
        )
        await builder.sync()

        return builder.directory(self._repaired_dir).file(
            whlname_maker.make_repaired_wheelname(python_target_version)
        )


class WheelName:
    """faiss-gpu wheel name provider."""

    def __init__(
        self, faiss_version: str, auditwheel_policy: str, cuda_major_version: str
    ) -> None:
        """constructor.

        Args:
            faiss_version: faiss python package version
            auditwheel_policy: auditwheel policy
            cuda_major_version: cuda major version
        """
        self._faiss_version = faiss_version
        self._auditwheel_policy = auditwheel_policy
        self._cuda_major_version = cuda_major_version
        self._arch = "x86_64"
        self._platform = "linux"

    def make_raw_wheelname(self, python_version: str) -> str:
        """Make raw wheel name.

        Args:
            python_version: python version {major}.{minor}

        Returns:
            raw wheel name
        """
        python_version = "cp" + python_version.replace(".", "")
        name_parts = [
            f"faiss_gpu_cu{self._cuda_major_version}",
            self._faiss_version,
            python_version,
            python_version,
            f"{self._platform}_{self._arch}",
        ]
        return "-".join(name_parts) + ".whl"

    def make_repaired_wheelname(self, python_version: str) -> str:
        """Make repaired wheel name.

        Args:
            python_version: python version {major}.{minor}

        Returns:
            repaired wheel name
        """
        python_version = "cp" + python_version.replace(".", "")
        if self._auditwheel_policy == "manlylinux2014":
            name_parts = [
                f"faiss_gpu_cu{self._cuda_major_version}",
                self._faiss_version,
                python_version,
                python_version,
                "manylinux_2_17",
                self._arch,
                f"{self._auditwheel_policy}_{self._arch}",
            ]
        elif self._auditwheel_policy == "manylinux_2_28":
            name_parts = [
                f"faiss_gpu_cu{self._cuda_major_version}",
                self._faiss_version,
                python_version,
                python_version,
                f"{self._auditwheel_policy}_{self._arch}",
            ]
        else:
            msg = f"Invalid auditwheel policy. {self._auditwheel_policy}"
            raise ValueError(msg)
        return "-".join(name_parts) + ".whl"
