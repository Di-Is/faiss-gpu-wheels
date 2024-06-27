"""Serve build setting.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .type import BuildType, InstructionSet


@dataclass
class Config:
    """Basic Config."""

    # faiss root relative path
    faiss_root: str = "faiss"

    @property
    def faiss_home(self) -> str:
        """Faiss install directory.

        Returns:
            faiss install directory
        """
        return os.getenv("FAISS_HOME", "/usr/local")

    @property
    def python_support_versions(self) -> list[str]:
        """Serve suport python versions.

        Returns:
            serve suport python versions
        """
        return os.getenv("PYTHON_SUPPORT_VERSIONS", "3.8;3.9;3.10;3.11;3.12").split(";")

    @property
    def instruction_set(self) -> InstructionSet:
        """Serve instruction set.

        Returns:
            instruction set value
        """
        env_value = os.getenv("FAISS_OPT_LEVEL", "generic").upper()
        if env_value == InstructionSet.GENERIC.name:
            inst_set = InstructionSet.GENERIC
        elif env_value == InstructionSet.AVX2.name:
            inst_set = InstructionSet.AVX2
        elif env_value == InstructionSet.AVX512.name:
            inst_set = InstructionSet.AVX512
        else:
            raise ValueError
        return inst_set

    @property
    def build_type(self) -> BuildType:
        """Serve faiss build type.

        Returns:
            faiss build type
        """
        enable_gpu = os.getenv("FAISS_ENABLE_GPU") == "ON"
        enable_raft = os.getenv("FAISS_ENABLE_RAFT") == "ON"
        if enable_raft and enable_gpu:
            build_type = BuildType.RAFT
        elif enable_gpu:
            build_type = BuildType.GPU
        else:
            build_type = BuildType.CPU
        return build_type


@dataclass
class GPUConfig:
    """GPU Config."""

    @property
    def cuda_major_version(self) -> str:
        """CUDA major version used to build faiss.

        Returns:
            CUDA major version
        """
        return os.getenv("CUDA_MAJOR_VERSION")

    @property
    def cuda_runtime_version(self) -> str:
        """CUDA runtime version used to build faiss.

        Returns:
            CUDA runtime version
        """
        return os.getenv("CUDA_RUNTIME_VERSION")

    @property
    def cublas_version(self) -> str:
        """CuBLAS version used to build faiss.

        Returns:
            cuBLAS version
        """
        return os.getenv("CUDA_CUBLAS_VERSION")

    @property
    def cuda_home(self) -> str:
        """CUDA install directory.

        Returns:
            CUDA install directory
        """
        return os.getenv("CUDA_HOME", "/usr/local/cuda")

    @property
    def dynamic_link(self) -> str:
        """CUDA dynamic link flag.

        Returns:
            CUDA dynamic link flag
        """
        return bool(os.getenv("CUDA_DYNAMIC_LINK", ""))
