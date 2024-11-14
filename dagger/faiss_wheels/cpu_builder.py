"""Serve faiss-cpu building function using dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

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

    def build(self) -> Container:
        """Build image using faiss-cpu wheel building.

        Returns:
            container for faiss-cpu wheel building
        """
        # build wheel builder image
        return self._host_directory.docker_build(
            dockerfile="./docker/Dockerfile.cpu",
            build_args=[
                BuildArg("BUILD_NJOB", self._build_config["njob"]),
                BuildArg("FAISS_OPT_LEVEL", self._build_config["instruction_set"]),
                BuildArg("AUDITWHEEL_POLICY", self._auditwheel_config["policy"]),
            ],
        )


class WheelBuilder(AbsWheelBuilder):
    """faiss-cpu wheel builder."""

    def __init__(
        self,
        ctr: Container,
        host_directory: Directory,
        build_config: BuildConfig,
        auditwheel_config: AuditWheelConfig,
        python_config: PythonConfig,
    ) -> None:
        """constructor.

        Args:
            ctr: container for faiss-gpu wheel building
            host_directory: repository root directory
            build_config: build config
            auditwheel_config: auditwheel config
            python_config: python config
        """
        self._ctr = ctr
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
        ctr = (
            self._ctr.with_directory("/project", self._host_directory)
            .with_workdir("/project")
            .with_env_variable(
                "PYTHON_SUPPORT_VERSIONS",
                ";".join(self._python_config["support_versions"]),
            )
            .with_env_variable("FAISS_OPT_LEVEL", self._build_config["instruction_set"])
            .with_env_variable("FAISS_ENABLE_GPU", "OFF")
            .with_env_variable("FAISS_ENABLE_RAFT", "OFF")
            .with_env_variable("UV_HTTP_TIMEOUT", "10000000")
            .with_exec(["uv", "build", "--wheel", "--python", python_target_version])
        )

        raw_wheel_name = (await ctr.directory("dist").glob("*.whl"))[0]
        ctr = ctr.with_exec(
            ["auditwheel", "repair", f"dist/{raw_wheel_name}"]
            + self._auditwheel_config["repair_option"]
        )

        repaired_wheel_name = (await ctr.directory("wheelhouse").glob("*.whl"))[0]
        return ctr.directory("wheelhouse").file(repaired_wheel_name)
