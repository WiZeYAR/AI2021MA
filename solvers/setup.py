from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("solvers/two_opt_with_candidate.pyx"),
)