"""Serve build setting

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import os
from typing import List
from dataclasses import dataclass

from .type import BuildType, InstructionSet


@dataclass
class Config:
    # faiss root relative path
    faiss_root: str = "faiss"
    # faiss install directory
    faiss_home: str = os.getenv("FAISS_HOME", "/usr/local")
    # enable gpu flag
    _enable_gpu = os.getenv("FAISS_ENABLE_GPU") == "ON"
    # enable raft flag
    _enable_raft = os.getenv("FAISS_ENABLE_RAFT") == "ON"
    # build target instruction set
    _inst_set = os.getenv("FAISS_OPT_LEVEL", "generic").upper()
    # support python version string (combined ;)
    _suport_python_versions = os.getenv(
        "PYTHON_SUPPORT_VERSIONS", "3.8;3.9;3.10;3.11;3.12"
    )

    @property
    def python_support_versions(self) -> List[str]:
        """serve suport python versions

        Returns:
            serve suport python versions
        """
        return self._suport_python_versions.split(";")

    @property
    def instruction_set(self) -> InstructionSet:
        """serve instruction set

        Returns:
            instruction set value
        """
        if self._inst_set == InstructionSet.GENERIC.name:
            inst_set = InstructionSet.GENERIC
        elif self._inst_set == InstructionSet.AVX2.name:
            inst_set = InstructionSet.AVX2
        elif self._inst_set == InstructionSet.AVX512.name:
            inst_set = InstructionSet.AVX512
        else:
            raise ValueError
        return inst_set

    @property
    def build_type(self) -> BuildType:
        """serve faiss build type

        Returns:
            faiss build type
        """

        if self._enable_raft and self._enable_gpu:
            build_type = BuildType.RAFT
        elif self._enable_gpu:
            build_type = BuildType.GPU
        else:
            build_type = BuildType.CPU
        return build_type


@dataclass
class GPUConfig:
    # CUDA major version used to build faiss
    cuda_major_version: str = os.getenv("CUDA_MAJOR_VERSION")
    # CUDA runtime version used to build faiss
    cuda_runtime_version: str = os.getenv("CUDA_RUNTIME_VERSION")
    # cuBLAS version used to build faiss
    cublas_version: str = os.getenv("CUDA_CUBLAS_VERSION")
    # CUDA install directory
    cuda_home: str = os.getenv("CUDA_HOME", "/usr/local/cuda")
    # CUDA dynamic link
    dynamic_link: bool = bool(os.getenv("CUDA_DYNAMIC_LINK", ""))
