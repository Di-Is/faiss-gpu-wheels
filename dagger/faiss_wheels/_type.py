"""Serve config dataclass in dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from typing import Literal

from pydantic import BaseModel, Field


class Cxx(BaseModel):
    """C++ config."""

    njob: int


class AuditWheel(BaseModel):
    """AuditWheel config."""

    repair_option: list[str] = Field(alias="repair-option")


class TestCase(BaseModel):
    """Python test config."""

    name: str
    image: str
    target_python_versions: list[str] = Field(alias="target-python-versions")
    install_command: list[str] = Field(alias="install-command")
    test_command: list[str] = Field(alias="test-command")


class Python(BaseModel):
    """Python config."""

    njob: int
    requires_python: str = Field(alias="requires-python")
    auditwheel: AuditWheel
    test: list[TestCase]


class CudaConfig(BaseModel):
    """Cuda config."""

    major: str
    minor: str
    patch: str
    target_archs: str = Field(alias="target-archs")


class Config(BaseModel):
    """Config."""

    image: str
    opt_level: str = Field(alias="opt-level")
    variant: Literal["cpu", "gpu-cu11", "gpu-cu12"]
    cxx: Cxx
    python: Python


class CpuConfig(Config):
    """Cpu config."""


class GpuCudaConfig(Config):
    """Gpu cuda config."""

    cuda: CudaConfig
