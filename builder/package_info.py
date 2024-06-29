"""Serve package information.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from setuptools.command.build_py import build_py

from .config import Config, GPUConfig
from .extension import ExtensionsFactory
from .type import BuildType
from .util import get_project_root

if TYPE_CHECKING:
    from setuptools import Extension


class PackageInfo:
    """Package infomation."""

    description: str = (
        "A library for efficient similarity search and clustering of dense vectors."
    )
    url: str = "https://github.com/Di-Is/faiss-gpu-wheels"
    author: str = "Di-Is"
    author_email: str = "rhoxbox@gmail.com@gmail.com"
    license: str = "MIT License"
    keywords: str = "search nearest neighbors"
    long_description_content_type: str = "text/markdown"

    @property
    def name(self) -> str:
        """Serve package name.

        Returns:
            package name
        """
        cfg = Config()
        if cfg.build_type == BuildType.CPU:
            name = "faiss-cpu"
        elif cfg.build_type == BuildType.GPU:
            gpu_cfg = GPUConfig()
            name = f"faiss-gpu-cu{gpu_cfg.cuda_major_version}"
            if not gpu_cfg.dynamic_link:
                name += "-static"
        elif cfg.build_type == BuildType.RAFT:
            gpu_cfg = GPUConfig()
            name = f"faiss-gpu-raft-cu{gpu_cfg.cuda_major_version}"
            if not gpu_cfg.dynamic_link:
                name += "-static"
        else:
            raise ValueError
        return name

    @property
    def version(self) -> str:
        """Package version.

        Returns:
            package version
        """
        version_path = Path(get_project_root()) / "version.txt"
        with version_path.open("r") as f:
            return f.read()

    @property
    def install_requires(self) -> list[str]:
        """Package dependencies.

        Returns:
            package dependencies
        """
        cfg = Config()
        gpu_cfg = GPUConfig()
        requires = ["numpy<2", "packaging"]
        if cfg.build_type in [BuildType.GPU, BuildType.RAFT] and gpu_cfg.dynamic_link:
            cuda_major = gpu_cfg.cuda_major_version
            requires += [
                f"nvidia-cuda-runtime-cu{cuda_major}>={gpu_cfg.cuda_runtime_version}",
                f"nvidia-cublas-cu{cuda_major}>={gpu_cfg.cublas_version}",
            ]
        return requires

    @property
    def extras_require(self) -> dict[str, list[str]]:
        """Package extra dependencies.

        Returns:
            package extra dependencies
        """
        cfg = Config()
        gpu_cfg = GPUConfig()
        extras = {}
        if cfg.build_type in [BuildType.GPU, BuildType.RAFT] and gpu_cfg.dynamic_link:
            cuda_major = gpu_cfg.cuda_major_version
            extras.update(
                fix_cuda=[
                    f"nvidia-cuda-runtime-cu{cuda_major}=={gpu_cfg.cuda_runtime_version}",
                    f"nvidia-cublas-cu{cuda_major}=={gpu_cfg.cublas_version}",
                ]
            )
        return extras

    @property
    def packages(self) -> list[str]:
        """Packaging directories.

        Returns:
            packaging directories
        """
        return ["faiss", "faiss.contrib"]

    @property
    def package_dir(self) -> dict[str, str]:
        """Packaging directories path.

        Returns:
            packaging directories path
        """
        faiss_root = Path(get_project_root()) / "faiss"
        return {
            "faiss": str(faiss_root / "faiss" / "python"),
            "faiss.contrib": str(faiss_root / "contrib"),
        }

    @property
    def classifiers(self) -> list[str]:
        """Classifiers wrote to METADATA.

        Returns:
            classifiers wrote to METADATA
        """
        classifiers = [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX",
        ]
        cfg = Config()
        classifiers += [
            f"Programming Language :: Python :: {ver}"
            for ver in cfg.python_support_versions
        ]
        # Add CUDA infomation
        if cfg.build_type in [BuildType.GPU, BuildType.RAFT]:
            gpu_cfg = GPUConfig()
            classifiers.append("Environment :: GPU :: NVIDIA CUDA")
            classifiers.append(
                f"Environment :: GPU :: NVIDIA CUDA :: {gpu_cfg.cuda_major_version}"
            )
        classifiers.append("Topic :: Scientific/Engineering :: Artificial Intelligence")

        return classifiers

    @property
    def long_description(self) -> str:
        """Package long description.

        Returns:
            package long description
        """
        readme_path = Path(get_project_root()) / "README.md"
        with readme_path.open("r") as f:
            return f.read()

    @property
    def include_package_data(self) -> bool:
        """Whether to package all non .py files.

        Returns:
            Whether to package all non .py files
        """
        return False

    @property
    def package_data(self) -> dict[str, list[str]]:
        """Data included in package.

        Returns:
            data included in package
        """
        return {"": ["*.i", "*.h", "TARGET_CUDA_MAJOR.txt"]}

    @property
    def ext_modules(self) -> list[Extension]:
        """Package extension modules.

        Returns:
            Package extension module.
        """
        cfg = Config()
        return ExtensionsFactory.generate(
            sys.platform, cfg.instruction_set, cfg.build_type
        )

    @property
    def cmdclass(self) -> dict[str, build_py]:
        """Custom build command.

        Returns:
            Custom build command
        """
        return {"build_py": CustomBuildPy}


class CustomBuildPy(build_py):
    """Run build_ext before build_py to compile swig code."""

    def run(self) -> None:
        """Execute build."""
        self.run_command("build_ext")
        return build_py.run(self)
