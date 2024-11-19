"""Serve dagger CI pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from typing import Annotated

import dagger
from dagger import dag, function, object_type
from faiss_wheels import _cpu_builder, _gpu_cuda_builder

from ._static_analysis import static_analysis
from ._test import execute_test
from ._util import execute_async_parallel, get_compatible_python, load_config


@object_type
class FaissWheels:
    """dagger ci piplline."""

    source: Annotated[dagger.Directory, dagger.Doc("reposiory root")]

    @function
    async def static_analysis(self) -> None:
        """Execute static analysis.

        Args:
            source: source directory
        """
        await static_analysis(self.source)

    @function
    async def build_container(self, config_file: dagger.File) -> dagger.Container:
        """Build image included faiss library.

        Args:
            config_file: config file.

        Returns:
            container for faiss building
        """
        config = await load_config(config_file)
        match config.variant:
            case "cpu":
                builder = _cpu_builder.ImageBuilder(self.source, config)
            case "gpu-cu11" | "gpu-cu12":
                builder = _gpu_cuda_builder.ImageBuilder(self.source, config)
            case _:
                raise NotImplementedError
        ctr = builder.build()
        return ctr.with_directory("/project", self.source).with_workdir("/project")

    @function
    async def build_wheels(self, config_file: dagger.File) -> dagger.Directory:
        """Build faiss wheels.

        Args:
            config_file: config file.

        Returns:
            container for faiss building
        """
        ctr = await self.build_container(config_file)
        config = await load_config(config_file)
        match config.variant:
            case "cpu":
                builder = _cpu_builder.WheelBuilder(ctr, config)
            case "gpu-cu11" | "gpu-cu12":
                builder = _gpu_cuda_builder.WheelBuilder(ctr, config)
            case _:
                raise NotImplementedError
        wheels = await execute_async_parallel(
            builder.build,
            [{"py_version": v} for v in get_compatible_python(config.python.requires_python)],
            config.python.njob,
        )
        return dag.directory().with_files(".", wheels)

    @function
    async def execute_pipeline(self, config_file: dagger.File) -> dagger.Directory:
        """Execute pipeline.

        Args:
            config_file: config file.

        Returns:
            directory included wheels
        """
        await self.static_analysis()
        wheel_dir = await self.build_wheels(config_file)
        await self.execute_test(wheel_dir, config_file)
        return wheel_dir

    @function
    async def execute_test(self, wheel_dir: dagger.Directory, config_file: dagger.File) -> None:
        """Execute test.

        Args:
            wheel_dir: directory included wheels
            config_file: config file
        """
        config = await load_config(config_file)
        await execute_test(self.source, wheel_dir, config)
