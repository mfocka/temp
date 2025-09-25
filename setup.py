"""
Setup script for ISM330DHCX Data Visualization Tool
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ism330dhcx-tool",
    version="1.0.0",
    author="ISM330DHCX Tool Development Team",
    author_email="",
    description="A Python-based tooling solution for real-time visualization and analysis of ISM330DHCX sensor data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ism330dhcx-tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ism330dhcx-tool=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="sensor, ism330dhcx, mems, visualization, serial, data-logging",
    project_urls={
        "Bug Reports": "https://github.com/your-org/ism330dhcx-tool/issues",
        "Source": "https://github.com/your-org/ism330dhcx-tool",
        "Documentation": "https://ism330dhcx-tool.readthedocs.io/",
    },
)
