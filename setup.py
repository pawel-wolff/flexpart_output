#!/usr/bin/env python

from setuptools import setup

setup(
    name='flexpart-output',
    version='0.1',
    description='FLEXPART postprocessing utilities',
    author='Pawel Wolff',
    author_email='pawel.wolff@aero.obs-mip.fr',
    packages=[
        'fpout',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'dask',
        'common',
        'xarray_extras',
    ],
)
