from setuptools import setup, find_packages

setup(
    name="muzic",
    version="0.1.0",
    packages=find_packages(),
    description="Microsoft's Music Understanding and Generation Library",
    author="Microsoft",
    author_email="",
    url="https://github.com/microsoft/muzic",
    install_requires=[
        "torch",
        "numpy",
        "miditoolkit",
        "transformers",
        "fairseq",
    ],
)