"""Serve misc function.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path


def get_project_root() -> str:
    """Get project root path.

    Returns:
        project root path
    """
    return Path(__file__).parent.parent
