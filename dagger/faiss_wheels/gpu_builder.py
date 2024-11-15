"""Serve faiss-gpu building function using dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import BuildArg, Container, Directory, File

from .builder import AbsWheelBuilder
from .type import AuditWheelConfig, BuildConfig, CUDAConfig, PythonConfig


class ImageBuilder:
    """image ctr for faiss-gpu."""

    def __init__(
        self,
        host_directory: Directory,
        build_config: BuildConfig,
        auditwheel_config: AuditWheelConfig,
        cuda_config: CUDAConfig,
    ) -> None:
        """constructor.

        Args:
            host_directory: repository root directory
            build_config: build config
            auditwheel_config: auditwheel config
            cuda_config: cuda config
        """
        self._host_directory = host_directory
        self._build_config = build_config
        self._auditwheel_config = auditwheel_config
        self._cuda_config = cuda_config

    def build(self) -> Container:
        """Build image using faiss-gpu wheel building.

        Returns:
            container for faiss-gpu wheel building
        """
        # build wheel ctr image
        return self._host_directory.docker_build(
            dockerfile="./docker/Dockerfile.gpu",
            build_args=[
                BuildArg("CUDA_MAJOR_VERSION", self._cuda_config["major_version"]),
                BuildArg("CUDA_MINOR_VERSION", self._cuda_config["minor_version"]),
                BuildArg("CUDA_ARCHITECTURES", self._cuda_config["architectures"]),
                BuildArg("BUILD_NJOB", self._build_config["njob"]),
                BuildArg("FAISS_OPT_LEVEL", self._build_config["instruction_set"]),
                BuildArg("AUDITWHEEL_POLICY", self._auditwheel_config["policy"]),
            ],
        )


class WheelBuilder(AbsWheelBuilder):
    """faiss-gpu wheel ctr."""

    def __init__(  # noqa: PLR0913
        self,
        ctr: Container,
        host_directory: Directory,
        build_config: BuildConfig,
        auditwheel_config: AuditWheelConfig,
        python_config: PythonConfig,
        cuda_config: CUDAConfig,
    ) -> None:
        """constructor.

        Args:
            ctr: container for faiss-gpu wheel building
            host_directory: repository root directory
            build_config: build config
            auditwheel_config: auditwheel config
            python_config: python config
            cuda_config: cuda config
        """
        self._ctr = ctr
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
        ctr = (
            self._ctr.with_directory("/project", self._host_directory)
            .with_workdir("/project")
            .with_env_variable("CUDA_MAJOR_VERSION", self._cuda_config["major_version"])
            .with_env_variable("CUDA_MINOR_VERSION", self._cuda_config["minor_version"])
            .with_env_variable("CUDA_RUNTIME_VERSION", self._cuda_config["runtime_version"])
            .with_env_variable("CUDA_CUBLAS_VERSION", self._cuda_config["cublas_version"])
            .with_env_variable(
                "CUDA_DYNAMIC_LINK", "ON" if self._cuda_config["dynamic_link"] else ""
            )
            .with_env_variable(
                "PYTHON_SUPPORT_VERSIONS", ";".join(self._python_config["support_versions"])
            )
            .with_env_variable("FAISS_OPT_LEVEL", self._build_config["instruction_set"])
            .with_env_variable("FAISS_ENABLE_GPU", "ON")
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
