"""Serve data structure type

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from typing import List, TypedDict
from enum import Enum


class BuildType(Enum):
    CPU: int = 0
    GPU: int = 1
    RAFT: int = 2


class BuildOption(TypedDict):
    name: str
    extra_compile_args: List[str]
    extra_link_args: List[str]
    swig_opts: List[str]


class InstructionSet(Enum):
    GENERIC: int = 0
    AVX2: int = 1
    AVX512: int = 2
