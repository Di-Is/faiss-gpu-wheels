"""package setup.py.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import sys
from pathlib import Path

from setuptools import setup

# Add project root to system path for import builder module.
sys.path.append(str(Path(__file__).parent))
from builder.package_info import PackageInfo
from builder.setup_preprocess import Preprocess

# Execute preprocess for package building
Preprocess.execute()

pkg_info = PackageInfo()

setup(
    name=pkg_info.name,
    version=pkg_info.version,
    description=pkg_info.description,
    long_description=pkg_info.long_description,
    long_description_content_type=pkg_info.long_description_content_type,
    url=pkg_info.url,
    author=pkg_info.author,
    author_email=pkg_info.author_email,
    license=pkg_info.license,
    keywords=pkg_info.keywords,
    install_requires=pkg_info.install_requires,
    extras_require=pkg_info.extras_require,
    packages=pkg_info.packages,
    package_dir=pkg_info.package_dir,
    package_data=pkg_info.package_data,
    classifiers=pkg_info.classifiers,
    ext_modules=pkg_info.ext_modules,
    cmdclass=pkg_info.cmdclass,
)
