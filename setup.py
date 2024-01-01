from setuptools import setup, find_packages

setup(
    name="VQMAP",
    version="0.1.0",
    author="",
    author_email="tianqing.li@duke.edu",
    description="Vector Quantized Representations for Efficient Hierarchical Delineation of Behavioral Repertoires",
    packages=find_packages(exclude=("configs", "deps")),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.11",
        "numpy",
        "tqdm",
        "matplotlib",
        "h5py",
        "munch"
    ],
)