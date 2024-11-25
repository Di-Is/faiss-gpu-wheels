"""Serve faiss-gpu-cuXX building function using dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import BuildArg, Container, Directory, File

from ._builder import AbsImageBuilder, AbsWheelBuilder
from ._type import GpuCudaConfig
from ._util import install_uv


class ImageBuilder(AbsImageBuilder):
    """image builder included faiss library."""

    def __init__(self, source: Directory, config: GpuCudaConfig) -> None:
        """Constructor.

        Args:
            source: repository root
            config: build target python version
        """
        self._source = source
        self._config = config

    def build(self) -> Container:
        """Build image using faiss-gpu wheel building.

        Returns:
            container for faiss-gpu wheel building
        """
        # build wheel ctr image
        return self._source.docker_build(
            dockerfile=f"./variant/faiss-{self._config.variant}/Dockerfile",
            build_args=[
                BuildArg("CUDA_MAJOR_VERSION", self._config.cuda.major),
                BuildArg("CUDA_MINOR_VERSION", self._config.cuda.minor),
                BuildArg("CUDA_PATCH_VERSION", self._config.cuda.patch),
                BuildArg("CUDA_ARCHITECTURES", self._config.cuda.target_archs),
                BuildArg("BUILD_NJOB", str(self._config.cxx.njob)),
                BuildArg("FAISS_OPT_LEVEL", self._config.opt_level),
                BuildArg("AUDITWHEEL_POLICY", self._config.image),
            ],
        )


class WheelBuilder(AbsWheelBuilder):
    """faiss-gpu-cuXX wheel builder."""

    def __init__(self, ctr: Container, config: GpuCudaConfig) -> None:
        """Constructor.

        Args:
            ctr: build target python version
            config: build target python version
        """
        self._ctr = ctr
        self._config = config

    async def build(self, py_version: str) -> File:
        """Build faiss-gpu wheel.

        Args:
            ctr: build target python version
            config: build target python version
            py_version: build target python version.

        Returns:
            faiss-gpu wheel file
        """
        ctr = (
            install_uv(self._ctr)
            .with_exec(["git", "apply", "patch/modify-numpy-find-package.patch"])
            .with_workdir(f"variant/faiss-{self._config.variant}")
            .with_exec(
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--python",
                    py_version,
                    "--out-dir",
                    ".",
                    "--config-setting",
                    f"cmake.define.FAISS_OPT_LEVEL={self._config.opt_level}",
                ]
            )
        )
        wheel_name = (await ctr.directory(".").glob("*.whl"))[0]
        ctr = ctr.with_exec(
            ["auditwheel", "repair", wheel_name, *self._config.python.auditwheel.repair_option]
        )
        repaired_wheel_name = (await ctr.directory("wheelhouse").glob("*.whl"))[0]
        return ctr.directory("wheelhouse").file(repaired_wheel_name)
