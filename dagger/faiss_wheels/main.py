"""Serve dagger CI pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from typing import Annotated

import dagger
from dagger import function, object_type

from ._static_analysis import static_analysis


@object_type
class FaissWheels:
    """dagger ci piplline."""

    source: Annotated[dagger.Directory, dagger.Doc("reposiory root")]

    @function
    async def static_analysis(self) -> None:
        """Execute static analysis.

        Args:
            source: source directory
        """
        await static_analysis(self.source)
