<div align="center">
<img src="https://github.com/OpenGSL/OpenGSL/blob/main/docs/source/img/opengsl.jpg" border="0" width=600px/>
</div>


------

<p align="center">
  <a href="#opengsl">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="https://github.com/OpenGSL/OpenGSL/tree/main/examples">Examples</a> •
  <a href="https://opengsl.readthedocs.io/en/latest/index.html">Docs</a> •
  <a href="#citation">Citation</a> 
</p>

[![Documentation Status](https://readthedocs.org/projects/opengsl/badge/?version=latest)](https://opengsl.readthedocs.io/en/latest/?badge=latest)
[![license](https://badgen.net/github/license/opengsl/opengsl?color=green)](https://github.com/opengsl/opengsl/blob/main/LICENSE)
![version](https://img.shields.io/badge/version-0.0.5-blue)


# OpenGSL
Official code for [OpenGSL: A Comprehensive Benchmark for Graph Structure Learning](https://arxiv.org/abs/2306.10280). OpenGSL is a comprehensive benchmark for **Graph Structure Learning(GSL)**. GSL is a family of data-centric learning approaches which jointly optimize the graph structure and the corresponding GNN models. It has great potential in many real-world applications, including disease analysis, protein structure prediction, etc.

## Overview of the Benchmark
OpenGSL provides a fair and comprehensive platform to evaluate existing GSL works and facilitate future GSL research.

![timeline](https://github.com/OpenGSL/OpenGSL/blob/main/docs/source/img/timeline.png)

## Installation
<!--
[PyTorch](https://pytorch.org/get-started/previous-versions/)
[PyTorch Geometric, PyTorch Sparse](https://data.pyg.org/whl/)
[DEEP GRAPH LIBRARY (DGL)](https://data.dgl.ai/wheels/repo.html)
-->
**Note:** OpenGSL depends on [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) and [DEEP GRAPH LIBRARY (DGL)](https://www.dgl.ai/pages/start.html). To streamline the installation, OpenGSL does **NOT** install these libraries for you. Please install them from the above links for running OpenGSL:

- torch>=1.9.1
- torch_geometric>=2.1.0
- torch_sparse>=0.6.12
- dgl>=0.9.0

**Installing with Pip**
``` bash
pip install opengsl
```

**Installation for local development:**
``` bash
git clone https://github.com/OpenGSL/OpenGSL
cd opengsl
pip install -e .
```

#### Required Dependencies:
- Python 3.7+
- ruamel.yaml
- pandas
- scipy
- scikit-learn
- pyro-api
- pyro-ppl
- numba


## Quick Start

You can use the command `python examples/gcn_cora.py` or follow the 4 steps.

The following example shows you how to perform GCN on the Cora dataset. 

#### Step 1: Load configuration

``` python
import opengsl
conf = opengsl.config.load_conf(method="gcn", dataset="cora")
```

##### Method and Dataset parameters in Built-in configuration

**method** ： 
`gcn`, `prognn`, `idgl`, `grcn`, `gaug`, `slaps`, `gt`,  `nodeformer`, `gen`, `cogsl`, `segsl`, `sublime`, `stable`

**dataset** ： 
`cora`, `pubmed`, `citeseer`, `blogcatalog`, `flickr`, `amazon-ratings`, `questions`,  `minesweeper`, `roman-empire`, `wiki-cooc`


#### Step 2: Load data
``` python
dataset = opengsl.data.Dataset("cora", n_splits=1, feat_norm=conf.dataset['feat_norm'])
```

#### Step 3: Build Model
``` python
solver = opengsl.method.GCNSolver(conf,dataset)
```

#### Step 4: Training and Evaluation
``` python
exp = opengsl.ExpManager(solver)
exp.run(n_runs = 10)
```

## Adding New GSL Method
if you want to use your own GSL method, see [customized_gsl.py](https://github.com/OpenGSL/OpenGSL/blob/main/examples/customized_gsl.py) for detail.

## Update
2023.9.26 A New GSL method GLCN added.

2023.9.15 New synthetic datasets to control the neighborhood pattern.

2023.8.14 New datasets including Wikics, Ogbn-arxiv and CSBM synthetic graphs.

2023.8.9 Enabling the use of APPNP and GIN as backbones for various GSL methods.

2023.7.31 A new GSL method WSGNN added.

2023.7.14 A scalable version of CoGSL added.

## Node Classification Results

### Results in Our Paper

| **Model**      |   **Cora**   | **Citeseer** |  **Pubmed**  | **Questions** | **Minesweeper** |
| -------------- | :----------: | :----------: | :----------: | :-----------: | :-------------: |
| **GCN**        | 81.95 ± 0.62 | 71.34 ± 0.48 | 78.98 ± 0.35 | 75.80 ± 0.51  |  78.28 ± 0.44   |
| **LDS**        | 84.13 ± 0.52 | 75.16 ± 0.43 |      --      |      --       |       --        |
| **ProGNN**     | 80.27 ± 0.48 | 71.35 ± 0.42 | 79.39 ± 0.29 |      --       |  51.43 ± 2.22   |
| **IDGL**       | 84.19 ± 0.61 | 73.26 ± 0.53 | 82.78 ± 0.44 | 50.00 ± 0.00  |  50.00 ± 0.00   |
| **GRCN**       | 84.61 ± 0.34 | 72.34 ± 0.73 | 79.30 ± 0.34 | 74.50 ± 0.84  |  72.57 ± 0.49   |
| **GAug**       | 83.43 ± 0.53 | 72.79 ± 0.86 | 78.73 ± 0.77 |      --       |  77.93 ± 0.64   |
| **SLAPS**      | 72.29 ± 1.01 | 70.00 ± 1.29 | 70.96 ± 0.99 |      --       |  50.89 ± 1.72   |
| **WSGNN**      | 83.66 ± 0.30 | 71.15 ± 1.01 | 79.78 ± 0.35 |      --       |  67.91 ± 3.11   |
| **Nodeformer** | 78.81 ± 1.21 | 70.39 ± 2.04 | 78.38 ± 1.94 | 72.61 ± 2.29  |  77.29 ± 1.71   |
| **GEN**        | 81.66 ± 0.91 | 73.21 ± 0.62 | 78.49 ± 3.98 |      --       |  79.56 ± 1.09   |
| **CoGSL**      | 81.46 ± 0.88 | 72.94 ± 0.71 | 78.38 ± 0.41 |      --       |       --        |
| **SEGSL**      | 81.04 ± 1.07 | 71.57 ± 0.40 | 79.26 ± 0.67 |      --       |       --        |
| **SUBLIME**    | 83.33 ± 0.73 | 72.44 ± 0.89 | 80.56 ± 1.32 | 67.21 ± 0.99  |  49.93 ± 1.36   |
| **STABLE**     | 83.25 ± 0.86 | 70.99 ± 1.19 | 81.46 ± 0.78 |      --       |  70.78 ± 0.27   |


| **Model**      | **BlogCatalog** |  **Flickr**  | **Amazon-ratings** | **Roman-empire** | **Wiki-cooc** |
| -------------- | :-------------: | :----------: | :----------------: | :--------------: | :-----------: |
| **GCN**        |  76.12 ± 0.42   | 61.60 ± 0.49 |    45.24 ± 0.29    |   70.41 ± 0.47   | 92.03 ± 0.19  |
| **LDS**        |  77.10 ± 0.27   |      --      |         --         |        --        |      --       |
| **ProGNN**     |  73.38 ± 0.30   | 52.88 ± 0.76 |         --         |   56.21 ± 0.58   | 89.07 ± 5.59  |
| **IDGL**       |  89.68 ± 0.24   | 86.03 ± 0.25 |    45.87 ± 0.58    |   47.10 ± 0.65   | 90.18 ± 0.27  |
| **GRCN**       |  76.08 ± 0.27   | 59.31 ± 0.46 |    50.06 ± 0.38    |   44.41 ± 0.41   | 90.59 ± 0.37  |
| **GAug**       |  76.92 ± 0.34   | 61.98 ± 0.67 |    48.42 ± 0.39    |   52.74 ± 0.48   | 91.30 ± 0.23  |
| **SLAPS**      |  91.73 ± 0.40   | 83.92 ± 0.63 |    40.97 ± 0.45    |   65.35 ± 0.45   | 89.09 ± 0.54  |
| **WSGNN**      |  92.30 ± 0.32   | 89.90 ± 0.19 |    42.36 ± 1.03    |   57.33 ± 0.69   | 90.10 ± 0.28  |
| **Nodeformer** |  44.53 ± 22.62  | 67.14 ± 6.77 |    41.33 ± 1.25    |   56.54 ± 3.73   | 54.83 ± 4.43  |
| **GEN**        |  90.48 ± 0.99   | 84.84 ± 0.81 |    49.17 ± 0.68    |        --        | 91.15 ± 0.49  |
| **CoGSL**      |  83.96 ± 0.54   | 75.10 ± 0.47 |    40.82 ± 0.13    |   46.52 ± 0.48  |      --       |
| **SEGSL**      |  75.03 ± 0.28   | 60.59 ± 0.54 |         --         |        --        |      --       |
| **SUBLIME**    |  95.29 ± 0.26   | 88.74 ± 0.29 |    44.49 ± 0.30    |   63.93 ± 0.27   | 76.10 ± 1.12  |
| **STABLE**     |  71.84 ± 0.56   | 51.36 ± 1.24 |    48.36 ± 0.21    |   41.00 ± 1.18   | 80.46 ± 2.44  |

### New Results

| **Model**      |    Wikics     |  Ogbn-arxiv  |
| :------------- | :-----------: | :----------: |
| **GCN**        | 78.83 ± 0.71  | 71.22 ± 0.43 |
| **LDS**        |      --       |      --      |
| **ProGNN**     |     77.85     |      --      |
| **IDGL**       | 78.83 ± 0.51  | 71.42 ± 0.33 |
| **GRCN**       | 79.46 ± 0.68  |      --      |
| **GAug**       |               |      --      |
| **SLAPS**      | 71.97 ± 0.94  | 56.60 ± 0.10 |
| **WSGNN**      | 79.11 ± 0.40  |      --      |
| **Nodeformer** | 55.25 ± 20.91 |    70.89     |
| **GEN**        | 79.33 ± 1.13  |      --      |
| **CoGSL**      | 79.53 ± 0.46  |      --      |
| **SEGSL**      |      --       |      --      |
| **SUBLIME**    | 79.34 ± 2.54  |    55.29     |
| **STABLE**     | 78.66 ± 0.25  |      --      |


## How to Contribute

As an active research topic, we are witnessing the rapid development of GSL methods.
Hence, this project will be frequently updated, and we welcome everyone interested in this topic to contribute! 

Please feel free to send PR or issue!

## Citation
Our paper on this benchmark will be released soon!

If you use our benchmark in your works, we would appreciate citations to the paper:

```bibtex
@article{zhou2023opengsl,
  title={OpenGSL: A Comprehensive Benchmark for Graph Structure Learning},
  author={Zhiyao Zhou, Sheng Zhou, Bochao Mao, Xuanyi Zhou, Jiawei Chen, Qiaoyu Tan, Daochen Zha, Can Wang, Yan Feng, Chun Chen},
  journal={arXiv preprint arXiv:2306.10280},
  year={2023}
}
```

## Reference

|**ID**| **Paper** | **Method** | **Conference** |
|---------|---------|:----------:|:--------------:|
|1| [Semi-supervised classification with graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf%EF%BC%89)      | GCN        | ICLR 2017 |
|2| [Learning Discrete Structures for Graph Neural Networks](https://arxiv.org/abs/1903.11960) | LDS | ICML 2019 |
|3| [Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049)  | ProGNN     | KDD 2020 |
|4| [Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf)  | IDGL       | NeurIPS 2020 |
|5| [Graph-Revised Convolutional Network](https://arxiv.org/pdf/1911.07123)  | GRCN       | ECML-PKDD 2020 |
|6| [Data Augmentation for Graph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17315/17122)  | GAug(O)    | AAAI 2021 |
|7| [SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks](https://proceedings.neurips.cc/paper/2021/file/bf499a12e998d178afd964adf64a60cb-Paper.pdf)  | SLAPS      | ICML 2021 |
|8| [Variational Inference for Training Graph Neural Networks in Low-Data Regime through Joint Structure-Label Estimation](https://dl.acm.org/doi/abs/10.1145/3534678.3539283) | WSGNN | KDD 2022 |
|9| [Nodeformer: A scalable graph structure learning transformer for node classification](https://proceedings.neurips.cc/paper_files/paper/2022/file/af790b7ae573771689438bbcfc5933fe-Paper-Conference.pdf)  | Nodeformer | NeurIPS 2022 |
|10| [Graph Structure Estimation Neural Networks](http://shichuan.org/doc/103.pdf)  | GEN        | WWW 2021 |
|11| [Compact Graph Structure Learning via Mutual Information Compression](https://arxiv.org/pdf/2201.05540)  | CoGSL      | WWW 2022 |
|12| [SE-GSL: A General and Effective Graph Structure Learning Framework through Structural Entropy Optimization](https://arxiv.org/pdf/2303.09778)  | SEGSL      | WWW 2023 |
|13| [Towards Unsupervised Deep Graph Structure Learning](https://arxiv.org/pdf/2201.06367)  | SUBLIME    | WWW 2022 |
|14| [Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN](https://dl.acm.org/doi/pdf/10.1145/3534678.3539484)  | STABLE     | KDD 2022 |

