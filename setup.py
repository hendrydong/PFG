import os
from setuptools import find_packages
from setuptools import setup


folder = os.path.dirname(__file__)
version_path = os.path.join(folder, "pfg", "version.py")

__version__ = None
with open(version_path) as f:
  exec(f.read(), globals())

req_path = os.path.join(folder, "requirements.txt")
install_requires = []
if os.path.exists(req_path):
  with open(req_path) as fp:
    install_requires = [line.strip() for line in fp]

readme_path = os.path.join(folder, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
  with open(readme_path) as fp:
    readme_contents = fp.read().strip()

setup(
    name="pfg",
    version=__version__,
    description="Preconditioned Funcional Gradient Flow for Sampling.",
    author="Hanze Dong",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    requires_python=">=3.7",
)