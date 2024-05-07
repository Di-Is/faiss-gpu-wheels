"""Serve abstract class for wheel builder

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from abc import ABCMeta, abstractmethod
import dagger


class AbsWheelBuilder(metaclass=ABCMeta):
    @abstractmethod
    async def build(python_version: str) -> dagger.File:
        """build wheel

        Args:
            python_version: python version {major}.{minor}

        Returns:
            wheel file
        """
