from setuptools import setup, find_packages

setup(
    name='csbev',
    description="Cross-species behavioral analysis",
    version="0.1.0",
    author="tianqingli",
    packages=find_packages(),
    python_requires=">=3.8",
    requires=(
        "torch",
        "matplotlib",
        "hydra-core",
        "omegaconf",
        "loguru",
        "wandb",
        "lightning",
        "scipy",
    ),
)