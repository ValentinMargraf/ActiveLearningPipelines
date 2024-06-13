import codecs
import os

import setuptools

NAME = "alpbench"
DESCRIPTION = "Active Learning Pipelines Benchmark"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/ValentinMargraf/ActiveLearningPipelines"
EMAIL = "valentin.margraf@ifi.lmu.de"
AUTHOR = "Valentin Margraf et al."
REQUIRES_PYTHON = ">=3.10.0"

work_directory = os.path.abspath(os.path.dirname(__file__))


# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    with codecs.open(str(os.path.join(work_directory, rel_path)), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delimiter = '"' if '"' in line else "'"
            return line.split(delimiter)[1]


with open(os.path.join(work_directory, "README.md"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(work_directory, "CHANGELOG.md"), encoding="utf-8") as f:
    changelog = f.read()

base_packages = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "py-experimenter",
    "mysql-connector-python",
    "openml",
    "scikit-learn",
    "scikit-activeml",
    "catboost",
    "xgboost",
]

full_packages = ["pytorch-tabnet", "tabpfn"]


setuptools.setup(
    name=NAME,
    version=get_version("alpbench/__init__.py"),
    description=DESCRIPTION,
    long_description="\n\n".join([readme, changelog]),
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls={
        "Tracker": "https://github.com/ValentinMargraf/ActiveLearningPipelines/issues",
        "Source": "https://github.com/ValentinMargraf/ActiveLearningPipelines",
        "Documentation": "https://activelearningpipelines.readthedocs.io/",
    },
    packages=setuptools.find_packages(include=("alpbench", "alpbench.*")),
    install_requires=base_packages,
    extras_require={"full": base_packages + full_packages},
    include_package_data=True,
    license="MIT",
    license_files=("LICENSE",),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=["python", "machine learning", "active learning", "benchmark", "tabular data", "classification"],
    zip_safe=True,
)
