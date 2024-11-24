"""Serve faiss building interface.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dagger import Container, File


class AbsImageBuilder(ABC):
    """Image builder included faiss library."""

    @abstractmethod
    def build(self, py_version: str) -> File:
        pass


class AbsWheelBuilder(ABC):
    """Wheel builder."""

    @abstractmethod
    def build(self) -> Container:
        pass
