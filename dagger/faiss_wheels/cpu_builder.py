"""Serve faiss-cpu building function using dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path

from dagger import BuildArg, Container, Directory, File

from .builder import AbsWheelBuilder
from .type import AuditWheelConfig, BuildConfig, PythonConfig


class ImageBuilder:
    """image builder for faiss-cpu."""

    def __init__(
        self,
        host_directory: Directory,
        build_config: BuildConfig,
        auditwheel_config: AuditWheelConfig,
    ) -> None:
        """constructor.

        Args:
            host_directory: host environment directory
            build_config: _description_
            auditwheel_config: _description_
        """
        self._host_directory = host_directory
        self._build_config = build_config
        self._auditwheel_config = auditwheel_config

    async def build(self) -> Container:
        """Build image using faiss-cpu wheel building.

        Returns:
            container for faiss-cpu wheel building
        """
        # build wheel builder image
        builder = self._host_directory.docker_build(
            dockerfile="./docker/Dockerfile.cpu",
            build_args=[
                BuildArg("BUILD_NJOB", self._build_config["njob"]),
                BuildArg("FAISS_OPT_LEVEL", self._build_config["instruction_set"]),
                BuildArg("AUDITWHEEL_POLICY", self._auditwheel_config["policy"]),
            ],
        )
        await builder.sync()
        return builder


class WheelBuilder(AbsWheelBuilder):
    """faiss-cpu wheel builder."""

    _raw_dir: str = "/output/raw_wheels"
    _repaired_dir = "/output/repaired_wheels"

    def __init__(
        self,
        builder: Container,
        host_directory: Directory,
        build_config: BuildConfig,
        auditwheel_config: AuditWheelConfig,
        python_config: PythonConfig,
    ) -> None:
        """constructor.

        Args:
            builder: container for faiss-gpu wheel building
            host_directory: repository root directory
            build_config: build config
            auditwheel_config: auditwheel config
            python_config: python config
        """
        self._builder = builder
        self._host_directory = host_directory
        self._build_config = build_config
        self._auditwheel_config = auditwheel_config
        self._python_config = python_config

    async def build(self, python_target_version: str) -> File:
        """Build faiss-cpu wheel.

        Args:
            python_target_version: build target python version

        Returns:
            faiss-cpu wheel file
        """
        # build faiss wheel
        builder = (
            self._builder.with_directory("/project", self._host_directory)
            .with_workdir("/project")
            .with_env_variable(
                "PYTHON_SUPPORT_VERSIONS",
                ";".join(self._python_config["support_versions"]),
            )
            .with_env_variable("FAISS_OPT_LEVEL", self._build_config["instruction_set"])
            .with_env_variable("FAISS_ENABLE_GPU", "OFF")
            .with_env_variable("FAISS_ENABLE_RAFT", "OFF")
            .with_exec(["mkdir", "-p", self._raw_dir, self._repaired_dir])
        )

        builder = builder.with_exec(
            [
                "uv",
                "build",
                "--wheel",
                "--python",
                f"python{python_target_version}",
                "--out-dir",
                self._raw_dir,
            ]
        )

        # get faiss version
        faiss_ver = await self._host_directory.file("version.txt").contents()
        faiss_ver = faiss_ver.replace("\n", "")

        whlname_maker = WheelName(faiss_ver, self._auditwheel_config["policy"])

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
    """faiss-cpu wheel name provider."""

    def __init__(self, faiss_version: str, auditwheel_policy: str) -> None:
        """constructor.

        Args:
            faiss_version: faiss python package version
            auditwheel_policy: auditwheel policy
        """
        self._faiss_version = faiss_version
        self._auditwheel_policy = auditwheel_policy
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
            "faiss_cpu",
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
        if self._auditwheel_policy == "manylinux2014":
            name_parts = [
                "faiss_cpu",
                self._faiss_version,
                python_version,
                python_version,
                f"manylinux_2_17_{self._arch}.{self._auditwheel_policy}_{self._arch}",
            ]
        elif self._auditwheel_policy == "manylinux_2_28":
            name_parts = [
                "faiss_cpu",
                self._faiss_version,
                python_version,
                python_version,
                f"{self._auditwheel_policy}_{self._arch}",
            ]
        else:
            msg = f"Invalid auditwheel policy. {self._auditwheel_policy}"
            raise ValueError(msg)
        return "-".join(name_parts) + ".whl"
