#!/usr/bin/env python3
"""A generalized Poisson solver for first-principles device simulations"""
DOCLINES = (__doc__ or "").split("\n")

import os
import sys

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

from setuptools import find_packages, setup

NAME = "qtgps"
VERSION = "0.5.3"
URL = "https://gitlab.ethz.ch/ggandus/poisson.git"

AUTHOR = "Guido Gandus"
EMAIL = "gandusgui@gmail.ch"

LICENSE = "Apache 2.0"


def readme():
    if not os.path.exists("README.md"):
        return ""
    return open("README.md", "r").read()


requires = {
    "python_requires": ">= " + "3.7",
    "install_requires": ["setuptools", "numpy >= " + "1.21", "scipy",],
    "setup_requires": ["numpy >= " + "1.21",],
}

metadata = dict(
    name="qtgps",
    description=DOCLINES[0],
    long_description=readme(),
    url="https://gitlab.ethz.ch/ggandus/qtgps.git",
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    platforms="any",
    test_suite="pytest",
    version=VERSION,
    zip_safe=False,
    **requires
)


if __name__ == "__main__":
    setup(**metadata)
