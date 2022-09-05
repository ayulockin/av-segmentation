import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="av-drivable-segmentation",
    version="0.0.1",
    author="Ayush Thakur, Soumik Rakshit",
    author_email="mein2work@gmail.com",
    description=(
        "Semantic Segmentation pipeline for BDD100K dataset in "
        "TensorFlow/Keras baked with Weights and Biases."
    ),
    license="MIT License",
    keywords="semantic_segmentation BDD100K tensorflow keras wandb",
    packages=["drivable", "configs"],
    long_description=read("README.md"),
)
