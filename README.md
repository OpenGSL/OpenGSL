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
![version](https://img.shields.io/badge/version-0.0.6-blue)


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


## 🚀Quick Start

You can use the command `python examples/gcn_cora.py` or follow the 4 steps.

The following example shows you how to run [GRCN](https://arxiv.org/abs/1911.07123) on the Cora dataset. 

#### Step 1: Load configuration

``` python
import opengsl
conf = opengsl.config.load_conf(method="grcn", dataset="cora")
```

##### Method and Dataset parameters in Built-in configuration (Updated Frequently)

**method** ： 
`gcn`, `prognn`, `idgl`, `grcn`, `gaug`, `slaps`, `nodeformer`, `gen`, `cogsl`, `segsl`, `sublime`, `stable`, `wsgnn`, `glcn`, `bmgcn`

**dataset** ： 
`cora`, `pubmed`, `citeseer`, `blogcatalog`, `flickr`, `amazon-ratings`, `questions`,  `minesweeper`, `roman-empire`, `wiki-cooc`, `wikics`


#### Step 2: Load data
``` python
dataset = opengsl.data.Dataset("cora", n_splits=1, feat_norm=conf.dataset['feat_norm'])
```

#### Step 3: Build Model
``` python
solver = opengsl.method.GRCNSolver(conf,dataset)
```

#### Step 4: Training and Evaluation
``` python
exp = opengsl.ExpManager(solver)
exp.run(n_runs = 10)
```

## ⚙️Build Your Own GSL
OpenGSL provides an easy way to build GSL algorithm based on several components.

if you want to learn how to build own GSL method, see [Build Your Own GSL.ipynb](https://github.com/OpenGSL/OpenGSL/blob/main/examples/Build%20Your%20Own%20GSL.ipynb) for more details.

## 📱️Updates
2023.11.1 Version 0.0.6 available!
* **A General GSL model** added. 
* New GSL methods WSGNN, GLCN and BMGCN added.
* New datasets including Wikics, Ogbn-arxiv and CSBM synthetic graphs.
* APPNP and GIN can be used as backbones for various GSL methods.
* Other minor updates.



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

| **ID** | **Paper** | **Method** | **Conference** |
|--------|---------|:----------:|:--------------:|
| 1      | [Semi-supervised classification with graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf%EF%BC%89)      |    GCN     |   ICLR 2017    |
| 2      | [Learning Discrete Structures for Graph Neural Networks](https://arxiv.org/abs/1903.11960) |    LDS     |   ICML 2019    |
| 3      | [Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049)  |   ProGNN   |    KDD 2020    |
| 4      | [Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf)  |    IDGL    |  NeurIPS 2020  |
| 5      | [Graph-Revised Convolutional Network](https://arxiv.org/pdf/1911.07123)  |    GRCN    | ECML-PKDD 2020 |
| 6      | [Data Augmentation for Graph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17315/17122)  |  GAug(O)   |   AAAI 2021    |
| 7      | [SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks](https://proceedings.neurips.cc/paper/2021/file/bf499a12e998d178afd964adf64a60cb-Paper.pdf)  |   SLAPS    |   ICML 2021    |
| 8      | [Variational Inference for Training Graph Neural Networks in Low-Data Regime through Joint Structure-Label Estimation](https://dl.acm.org/doi/abs/10.1145/3534678.3539283) |   WSGNN    |    KDD 2022    |
| 9      | [Nodeformer: A scalable graph structure learning transformer for node classification](https://proceedings.neurips.cc/paper_files/paper/2022/file/af790b7ae573771689438bbcfc5933fe-Paper-Conference.pdf)  | Nodeformer |  NeurIPS 2022  |
| 10     | [Graph Structure Estimation Neural Networks](http://shichuan.org/doc/103.pdf)  |    GEN     |    WWW 2021    |
| 11     | [Compact Graph Structure Learning via Mutual Information Compression](https://arxiv.org/pdf/2201.05540)  |   CoGSL    |    WWW 2022    |
| 12     | [SE-GSL: A General and Effective Graph Structure Learning Framework through Structural Entropy Optimization](https://arxiv.org/pdf/2303.09778)  |   SEGSL    |    WWW 2023    |
| 13     | [Towards Unsupervised Deep Graph Structure Learning](https://arxiv.org/pdf/2201.06367)  |  SUBLIME   |    WWW 2022    |
| 14     | [Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN](https://dl.acm.org/doi/pdf/10.1145/3534678.3539484)  |   STABLE   |    KDD 2022    |
| 15     | [Semi-Supervised Learning With Graph Learning-Convolutional Networks](https://ieeexplore.ieee.org/document/8953909/authors#authors)      |    GLCN    |   CVPR 2019    |   
| 15     | [Block Modeling-Guided Graph Convolutional Neural Networks](http://arxiv.org/abs/2112.13507)      |   BM-GCN   |   AAAI 2022    |

