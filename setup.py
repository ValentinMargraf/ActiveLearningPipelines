import os

import setuptools

from ALP import (
    author,
    author_email,
    description,
    package_name,
    project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        # Dependencies
        "scikit-learn>=1.2.0",
        "scikit-activeml>=0.4.1",
        "py-experimenter>=1.4.1",
        "mysql-connector-python>=8.3.0",
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Docs
        "automl_sphinx_theme",
        # Others
        "isort",
        "black",
        "pydocstyle",
        "flake8",
        "pre-commit",
    ]
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    url=url,
    project_urls=project_urls,
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy"
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
