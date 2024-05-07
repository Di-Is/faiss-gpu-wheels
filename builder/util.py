"""Serve misc function

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import os


def get_project_root() -> str:
    """get project root path

    Returns:
        project root path
    """
    return os.path.dirname(os.path.dirname(__file__))
