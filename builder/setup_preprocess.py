"""Serve wheel build preprocess function.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import Config, GPUConfig
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

    cuda_major_file: str = "TARGET_CUDA_MAJOR.txt"

    @classmethod
    def execute(cls) -> None:
        """Execute preprocessing before package build."""
        # When static link is used, insertion of CUDA preloader is not necessary
        if not GPUConfig().dynamic_link:
            return

        cls._write_cuda_major_version()
        cls._insert_cuda_preloader_trigger_to_faiss()
        cls._copy_cuda_preloader_to_faiss()

    @classmethod
    def _write_cuda_major_version(cls) -> None:
        """Write cuda major version to text file in faiss package."""
        root = get_project_root()
        cfg = Config()
        gpu_cfg = GPUConfig()

        # write cuda major version to faiss python package directory
        cuda_major_txt_path = (
            Path(root) / cfg.faiss_root / "faiss" / "python" / cls.cuda_major_file
        )

        with cuda_major_txt_path.open("w") as f:
            f.write(gpu_cfg.cuda_major_version)

    @classmethod
    def _insert_cuda_preloader_trigger_to_faiss(cls) -> None:
        """Insert cure_prelaoder.py import line to faiss.loader.py."""
        trigger_str = '''
###################################################
"""
Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""
# Add faiss-wheels patch
import faiss.cuda_preloader
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
    def _copy_cuda_preloader_to_faiss(cls) -> None:
        """Copy cuda_preloader.py to faiss package."""
        cfg = Config()
        src_path = str(Path(Path(__file__).parent) / "faiss" / "cuda_preloader.py")
        dest_path = str(Path(get_project_root()) / cfg.faiss_root / "faiss" / "python")
        shutil.copy(src_path, dest_path)


class RAFTPreProcess:
    """Preprocess for raft build."""

    @classmethod
    def execute(cls) -> None:
        """Execute preprocessing before package build."""
