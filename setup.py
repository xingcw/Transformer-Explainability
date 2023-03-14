import setuptools

__version__ = '0.0.1'

pkgs = setuptools.find_packages()

setuptools.setup(
    name='trans_exp',
    version=__version__,
    packages=pkgs,
)