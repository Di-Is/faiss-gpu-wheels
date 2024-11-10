"""Serve abstract class for wheel builder.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from abc import ABCMeta, abstractmethod

import dagger


class AbsWheelBuilder(metaclass=ABCMeta):
    """Abstract class for wheel builder."""

    @abstractmethod
    async def build(self: str) -> dagger.File:
        """Build wheel.

        Args:
            python_version: python version {major}.{minor}

        Returns:
            wheel file
        """
