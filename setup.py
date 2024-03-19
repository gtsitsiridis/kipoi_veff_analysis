#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "cyvcf2",
    # "openpyxl",
    "numpy",
    "pandas",
    "pyarrow",
    "pyranges",
    "kipoiseq"
]

test_requirements = [
    "pytest",
]


setup(
    name='kipoi-enformer',
    version='0.0.1',
    description="Kipoi-based Enformer variant effect prediction",
    author="Florian Hölzlwimmer",
    author_email='git.ich@frhoelzlwimmer.de',
    url='https://gitlab.cmm.in.tum.de/hoelzlwi/kipoi-enformer',
    long_description="",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": ["bumpversion",
                    "wheel",
                    "jedi",
                    "epc",
                    "pytest",
                    "pytest-pep8",
                    "pytest-cov"],
    },
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    # package_data={'kipoi-enformer': ['logging.conf']},
    tests_require=test_requirements
)
