"""Serve data structure type.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class BuildType(Enum):
    """Faiss build type."""

    CPU: int = 0
    GPU: int = 1
    RAFT: int = 2


class BuildOption(TypedDict):
    """Faiss build option."""

    name: str
    extra_compile_args: list[str]
    extra_link_args: list[str]
    swig_opts: list[str]


class InstructionSet(Enum):
    """Faiss build instruction set."""

    GENERIC: int = 0
    AVX2: int = 1
    AVX512: int = 2
