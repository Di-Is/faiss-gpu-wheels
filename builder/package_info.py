"""Serve package information

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import sys
import os
from typing import List, Dict

from setuptools import Extension
from setuptools.command.build_py import build_py

from .extension import ExtensionsFactory
from .config import Config, GPUConfig
from .type import BuildType
from .util import get_project_root


class PackageInfo:
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
        """serve package name

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
        """package version

        Returns:
            package version
        """
        project_root = get_project_root()
        with open(f"{project_root}/version.txt", "r") as f:
            version = f.read()
        return version

    @property
    def install_requires(self) -> List[str]:
        """package dependencies

        Returns:
            package dependencies
        """
        cfg = Config()
        gpu_cfg = GPUConfig()
        requires = ["numpy", "packaging"]
        if cfg.build_type in [BuildType.GPU, BuildType.RAFT] and gpu_cfg.dynamic_link:
            cuda_major = gpu_cfg.cuda_major_version
            requires += [
                f"nvidia-cuda-runtime-cu{cuda_major}>={gpu_cfg.cuda_runtime_version}",
                f"nvidia-cublas-cu{cuda_major}>={gpu_cfg.cublas_version}",
            ]
        return requires

    @property
    def extras_require(self) -> Dict[str, List[str]]:
        """package extra dependencies

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
    def packages(self) -> List[str]:
        """packaging directories

        Returns:
            packaging directories
        """
        return ["faiss", "faiss.contrib"]

    @property
    def package_dir(self) -> Dict[str, str]:
        """packaging directories path

        Returns:
            packaging directories path
        """
        faiss_root = os.path.join(get_project_root(), "faiss")
        package_dir = {
            "faiss": os.path.join(faiss_root, "faiss", "python"),
            "faiss.contrib": os.path.join(faiss_root, "contrib"),
        }
        return package_dir

    @property
    def classifiers(self) -> List[str]:
        """classifiers wrote to METADATA

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
        """package long description

        Returns:
            package long description
        """
        root = get_project_root()
        with open(os.path.join(root, "README.md")) as f:
            return f.read()

    @property
    def include_package_data(self) -> bool:
        """Whether to package all non .py files

        Returns:
            Whether to package all non .py files
        """
        return False

    @property
    def package_data(self) -> Dict[str, List[str]]:
        """data included in package

        Returns:
            data included in package
        """
        return {"": ["*.i", "*.h", "TARGET_CUDA_MAJOR.txt"]}

    @property
    def ext_modules(self) -> List[Extension]:
        cfg = Config()
        return ExtensionsFactory.generate(
            sys.platform, cfg.instruction_set, cfg.build_type
        )

    @property
    def cmdclass(self):
        return {"build_py": CustomBuildPy}


class CustomBuildPy(build_py):
    """Run build_ext before build_py to compile swig code."""

    def run(self):
        self.run_command("build_ext")
        return build_py.run(self)
