import setuptools
from pip._internal.req import parse_requirements


install_reqs = parse_requirements("requirements.txt", session="test")
reqs = [str(ir.requirement) for ir in install_reqs]

__version__ = '0.0.1'

pkgs = setuptools.find_packages()

pkg_dirs = {}
new_pkgs = []

for pkg in pkgs:
    
    pkg_dir = "./" + "/".join(pkg.split("."))
    
    if "trans_exp" not in pkg:
        pkg = "trans_exp." + pkg
        
    new_pkgs.append(pkg)
    pkg_dirs.update({pkg: pkg_dir})

print(pkgs)
print(pkg_dirs)

setuptools.setup(
    name='trans_exp',
    version=__version__,
    packages=new_pkgs,
    package_dir=pkg_dirs,
    # Only put dependencies that's not depends on cuda directly.
    install_requires=reqs
)