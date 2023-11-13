import pathlib
from setuptools import setup, find_packages
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="rware",
    version="1.0.3",
    description="Multi-Robot Warehouse environment for reinforcement learning",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Filippos Christianos",
    url="https://github.com/semitable/robotic-warehouse",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy>=1.23.1",
        "gymnasium>=0.26.2",
        "pettingzoo>=1.23.0",
        "pillow",
        "networkx",
    ],
    extras_require={
        "test": ["pytest"],
        "full_render": ["pyglet<2"],
    },
    include_package_data=True,
)
