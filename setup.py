import setuptools
from pip._internal.req import parse_requirements


install_reqs = parse_requirements("requirements.txt", session="test")
reqs = [str(ir.requirement) for ir in install_reqs]

__version__ = '0.0.1'

pkgs = setuptools.find_packages()

setuptools.setup(
    name='trans_exp',
    version=__version__,
    packages=pkgs,
    install_requires=reqs
)