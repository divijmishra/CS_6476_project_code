from setuptools import setup, find_packages

setup(
    name="cs6476_project",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)