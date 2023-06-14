from setuptools import setup, find_packages

REQUIRES = """
ruamel.yaml
pandas
scipy
scikit-learn
wandb
pyro-api
pyro-ppl
numpy
numba >= 0.57.0
torch >= 1.12.1
torch_geometric >= 2.3.0
torch_sparse >= 0.6.17
dgl >= 1.0.2
"""


def get_install_requires():
    reqs = [req for req in REQUIRES.split("\n") if len(req) > 0]
    return reqs


with open("README.md") as f:
    readme = f.read()


def do_setup():
    setup(
        name="opengsl",
        version="0.0.1",
        description="A comprehensive benchmark for Graph Structure Learning.",
        url="https://github.com/OpenGSL/OpenGSL",
        author='Zhiyao Zhou, Sheng Zhou, Bochao Mao, Xuanyi Zhou',
        long_description=readme,
        long_description_content_type="text/markdown",
        install_requires=get_install_requires(),
        python_requires=">=3.10.0",
        packages=find_packages(
            include=["*"],
        ),
        keywords=["AI", "GNN", "graph structure learning"],
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
        ]
    )


if __name__ == "__main__":
    do_setup()