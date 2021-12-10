import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

PKG_NAME = "flyhostel"

# This call to setup() does all the work
setup(
    name=PKG_NAME,
    version="1.0.0",
    #description="High resolution monitoring of Drosophila",
    #long_description=README,
    #long_description_content_type="text/markdown",
    ##url="https://github.com/realpython/reader",
    #author="Antonio Ortega",
    #author_email="antonio.ortega@kuleuven.be",
    #license="MIT",
    #classifiers=[
    #    "License :: OSI Approved :: MIT License",
    #    "Programming Language :: Python :: 3",
    #    "Programming Language :: Python :: 3.7",
    #],
    packages = find_packages(),
    #include_package_data=True,
    install_requires=["pyserial"],
    entry_points={
        "console_scripts": [
            "fh=flyhostel:main"
            ]
    },
)
