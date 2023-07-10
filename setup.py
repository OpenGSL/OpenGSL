from setuptools import setup, find_packages

REQUIRES = """
ruamel.yaml
pandas
scipy
scikit-learn
pyro-api==0.1.2
pyro-ppl==1.8.0
numba
"""

def get_install_requires():
    reqs = [req for req in REQUIRES.split("\n") if len(req) > 0]
    return reqs


with open("README.md", encoding="utf-8") as f:
    readme = f.read()


def do_setup():
    setup(
        name="opengsl",
        version="0.0.4",
        description="A comprehensive benchmark for Graph Structure Learning.",
        url="https://github.com/OpenGSL/OpenGSL",
        author='Zhiyao Zhou, Sheng Zhou, Bochao Mao, Xuanyi Zhou',
        long_description=readme,
        long_description_content_type="text/markdown",
        install_requires=get_install_requires(),
        python_requires=">=3.7.0",
        packages=find_packages(),
        include_package_data=True,
        keywords=["AI", "GNN", "graph structure learning"],
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
        ]
    )


if __name__ == "__main__":
    do_setup()