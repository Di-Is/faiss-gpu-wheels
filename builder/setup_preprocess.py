"""Serve wheel build preprocess function.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import Config, GPUConfig
from .faiss.shared_library_preloader import PackageFileSearchArg
from .type import BuildType
from .util import get_project_root


class Preprocess:
    """Execute preprocessing before package build."""

    @classmethod
    def execute(cls) -> None:
        """Execute preprocessing before package build depending on build_type."""
        cfg = Config()
        CommonPreProcess.execute()
        if cfg.build_type == BuildType.CPU:
            CPUPreProcess.execute()
        elif cfg.build_type == BuildType.GPU:
            GPUPreProcess.execute()
        elif cfg.build_type == BuildType.RAFT:
            RAFTPreProcess.execute()


class CommonPreProcess:
    """Preprocess common to all build types."""

    @classmethod
    def execute(cls) -> None:
        """Execute preprocess."""
        cfg = Config()
        root = get_project_root()
        shutil.copy(
            str(Path(root) / cfg.faiss_root / "faiss" / "python" / "swigfaiss.swig"),
            str(Path(root) / cfg.faiss_root / "faiss" / "python" / "swigfaiss.i"),
        )


class CPUPreProcess:
    """Preprocess for cpu build."""

    @classmethod
    def execute(cls) -> None:
        """Execute preprocessing before package build."""


class GPUPreProcess:
    """Preprocess for gpu build."""

    preload_shared_libraries_file: str = "PRELOAD_SHARED_LIBRARIES.json"

    @classmethod
    def execute(cls) -> None:
        """Execute preprocessing before package build."""
        # When static link is used, insertion of CUDA preloader is not necessary
        if not GPUConfig().dynamic_link:
            return

        cls._write_preload_shared_libraries_file()
        cls._insert_shared_library_preloader_trigger()
        cls._copy_shared_library_preloader()

    @classmethod
    def _write_preload_shared_libraries_file(cls) -> None:
        """Write preloading shared libs json in faiss package."""
        root = get_project_root()
        cfg = Config()
        gpu_cfg = GPUConfig()

        json_path = (
            Path(root)
            / cfg.faiss_root
            / "faiss"
            / "python"
            / cls.preload_shared_libraries_file
        )

        # write preload target shared libraries json to faiss python package directory
        with json_path.open("w") as f:
            # preload cuda runtime and cublas
            json.dump(
                [
                    PackageFileSearchArg(
                        package_name=f"nvidia-cuda-runtime-cu{gpu_cfg.cuda_major_version}",
                        filename_regex=f"libcudart.so.{gpu_cfg.cuda_major_version}*",
                        group="CUDA",
                    ),
                    PackageFileSearchArg(
                        package_name=f"nvidia-cublas-cu{gpu_cfg.cuda_major_version}",
                        filename_regex=f"libcublas.so.{gpu_cfg.cuda_major_version}",
                        group="CUDA",
                    ),
                ],
                f,
            )

    @classmethod
    def _insert_shared_library_preloader_trigger(cls) -> None:
        """Insert shared_library_preloader.py import line to faiss.loader.py."""
        trigger_str = '''
###################################################
"""
Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""
# Add faiss-wheels patch
import faiss.shared_library_preload_trigger
###################################################

'''
        root = get_project_root()
        cfg = Config()
        loader_path = str(
            Path(root) / cfg.faiss_root / "faiss" / "python" / "loader.py"
        )
        back_path = loader_path.replace("loader.py", "loader.py.original")

        if Path(back_path).exists():
            return

        shutil.copy(loader_path, back_path)
        with Path(loader_path).open("+r") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(trigger_str + os.linesep + content)

    @classmethod
    def _copy_shared_library_preloader(cls) -> None:
        """Copy shared library preloader to faiss package."""
        cfg = Config()

        for src_path in (Path(__file__).parent / "faiss").glob("*.py"):
            dest_path = str(
                Path(get_project_root()) / cfg.faiss_root / "faiss" / "python"
            )
            shutil.copy(src_path, dest_path)


class RAFTPreProcess:
    """Preprocess for raft build."""

    @classmethod
    def execute(cls) -> None:
        """Execute preprocessing before package build."""
