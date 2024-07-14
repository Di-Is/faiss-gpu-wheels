"""Serve datastructure in dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from typing import TypedDict


class AuditWheelConfig(TypedDict):
    """auditwheel config."""

    policy: str
    repair_option: str


class PythonConfig(TypedDict):
    """python config."""

    support_versions: str


class CUDAConfig(TypedDict):
    """CUDA config."""

    major_version: str
    minor_version: str
    runtime_version: str
    cublas_version: str
    architectures: str
    dynamic_link: bool


class BuildConfig(TypedDict):
    """Faiss build config."""

    instruction_set: str
    enable_gpu: bool
    enable_raft: bool
    njob: int


class RAFTConfig(TypedDict):
    """RAFT config."""

    version: str
