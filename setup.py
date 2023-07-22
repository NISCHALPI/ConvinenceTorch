""""Setup for package."""
from setuptools import find_packages, setup

# Load the README.md file for long_description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="convinence-torch",
    version="1.0.0",
    description="Minimalistic convenience utilities for using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nischal Bhattarai",
    author_email="nischalbhattaraipi@gmail.com",
    url="https://github.com/NISCHALPI/ConvinenceTorch",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    keywords=["torch", "utilities", "trainer", "neural network"],
    python_requires=">=3.8",
    packages=find_packages('src'),
    package_dir={"":"src"},
    install_requires=[
        "torch >= 2.0.0",
        "scikit-learn >= 1.0",
        "matplotlib",
        "tqdm",
        "pandas",
    ],
    extras_require={
        "dev": ["nvitop", "ruff", "black", "mypy", "pytest"],
        "doc": ["sphinx", "myst-parser", "sphinx_rtd_theme", "nbsphinx"],
    },
    project_urls={
        "Homepage": "https://github.com/NISCHALPI/ConvinenceTorch",
    },
)
