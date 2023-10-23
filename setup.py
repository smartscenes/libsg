#!/usr/bin/env python

import runpy
from skbuild import setup  # from pyproject.toml


__version__ = runpy.run_path("libsg/__init__.py")["__version__"]


setup(
    name="libsg",
    version=__version__,
    author="3dlg-hcvc",
    url="https://github.com/smartscenes/libsg",
    description="libSG: Scene Generation Library",
    packages=["libsg", "libsg.vhacd"],
    cmake_with_sdist=True,
    # cmake_source_dir="cpp",  # uncomment to enable native code build
    cmake_install_dir="libsg",
    include_package_data=True,
    python_requires=">=3.9"
)
