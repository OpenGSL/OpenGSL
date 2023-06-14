# OpenGSL
OpenGSL is a comprehensive benchmark for Graph Structure Learning.

## Overview of the benchmark
OpenGSL is a easy interface for Graph Structure Learning.

Only the GPU version is currently available.


## Installation
Note: OpenGSL requires Python 3.7+

**Using Pip**
``` bash
pip install opengsl
```

**Installation for local development:**
``` bash
git clone https://github.com/OpenGSL/OpenGSL
cd opengsl
pip install -e .
```

## Quick Start
Following example shows you how to perform GCN on Cora dataset. 

#### Step 1: Load configuration
``` python
import opengsl
conf = opengsl.load_conf(method="gcn", dataset="cora")
```

#### Step 2: Load data
``` python
dataset = opengsl.data.Dataset("cora", n_splits=1, feat_norm=conf.dataset['feat_norm'])
```

#### Step 3: Build Model
``` python
solver = opengsl.method.gcn(conf,dataset)
```

#### Step 4: Training and Evaluation
``` python
exp = opengsl.ExpManager(solver, n_runs = 10)
exp.run()
```

## Add method
if you want to use your own method, see `example.py` for detail.


## Node Classification Results
| **Model**      | **Cora**     | **Citeseer** | **Pubmed**   | **Questions** | **Minesweeper** |
|----------------|:------------:|:------------:|:------------:|:-------------:|:---------------:|
| **GCN**        | 81.95 ± 0.62 | 71.34 ± 0.48 | 78.98 ± 0.35 | 75.80 ± 0.51  | 78.28 ± 0.44    |
| **ProGNN**     | 80.27 ± 0.48 | 71.35 ± 0.42 | 79.39 ± 0.29 | --            | 51.43 ± 2.22    |
| **IDGL**       | 84.19 ± 0.61 | 73.26 ± 0.53 | 82.78 ± 0.44 | 50.00 ± 0.00  | 50.00 ± 0.00    |
| **GRCN**       | 84.61 ± 0.34 | 72.34 ± 0.73 | 79.30 ± 0.34 | 74.50 ± 0.84  | 72.57 ± 0.49    |
| **GAug(O)**    | 83.43 ± 0.53 | 72.79 ± 0.86 | 78.73 ± 0.77 | --            | 77.93 ± 0.64    |
| **SLAPS**      | 72.29 ± 1.01 | 70.00 ± 1.29 | 70.96 ± 0.99 | --            | 50.89 ± 1.72    |
| **GT**         | 80.79 ± 1.14 | 68.50 ± 2.07 | 77.91 ± 0.58 | 75.36 ± 0.92  | 89.70 ± 0.28    |
| **Nodeformer** | 78.81 ± 1.21 | 70.39 ± 2.04 | 78.38 ± 1.94 | 72.61 ± 2.29  | 77.29 ± 1.71    |
| **GEN**        | 81.66 ± 0.91 | 73.21 ± 0.62 | 78.49 ± 3.98 | --            | 79.56 ± 1.09    |
| **CoGSL**      | 81.46 ± 0.88 | 72.94 ± 0.71 | --           | --            | --              |
| **SEGSL**      | 81.04 ± 1.07 | 71.57 ± 0.40 | 79.26 ± 0.67 | --            | --              |
| **SUBLIME**    | 83.33 ± 0.73 | 72.44 ± 0.89 | 80.56 ± 1.32 | 67.21 ± 0.99  | 49.93 ± 1.36    |
| **STABLE**     | 83.25 ± 0.86 | 70.99 ± 1.19 | 81.46 ± 0.78 | --            | 70.78 ± 0.27    |


| **Model**      | **BlogCatalog** | **Flickr**   | **Amazon-ratings** | **Roman-empire** | **Wiki-cooc**  |
|----------------|:---------------:|:------------:|:------------------:|:----------------:|:--------------:|
| **GCN**        | 76.12 ± 0.42    | 61.60 ± 0.49 | 45.24 ± 0.29       | 70.41 ± 0.47     | 92.03 ± 0.19   |
| **ProGNN**     | 73.38 ± 0.30    | 52.88 ± 0.76 | --                 | 56.21 ± 0.58     | 89.07 ± 5.59   |
| **IDGL**       | 89.68 ± 0.24    | 86.03 ± 0.25 |  45.87 ± 0.58      | 47.10 ± 0.65     |  90.18 ± 0.27  |
| **GRCN**       | 76.08 ± 0.27    | 59.31 ± 0.46 | 50.06 ± 0.38       | 44.41 ± 0.41     | 90.59 ± 0.37   |
| **GAug(O)**    | 76.92 ± 0.34    | 61.98 ± 0.67 | 48.42 ± 0.39       | 52.74 ± 0.48     | 91.30 ± 0.23   |
| **SLAPS**      | 91.73 ± 0.40    | 83.92 ± 0.63 | 40.97 ± 0.45       | 65.35 ± 0.45     | 89.09 ± 0.54   |
| **GT**         | 70.70 ± 5.62    | 43.19 ± 6.53 | 48.55 ± 0.34       | 76.49 ± 0.80     | 90.26 ± 1.24   |
| **Nodeformer** | 44.53 ± 22.62   | 67.14 ± 6.77 | 41.33 ± 1.25       | 56.54 ± 3.73     | 54.83 ± 4.43   |
| **GEN**        | 90.48 ± 0.99    | 84.84 ± 0.81 | 49.17 ± 0.68       | --               | 91.15 ± 0.49   |
| **CoGSL**      | --              | --           | --                 | --               | --             |
| **SEGSL**      | 75.03 ± 0.28    | 60.59 ± 0.54 | --                 | --               | --             |
| **SUBLIME**    | 95.29 ± 0.26    | 88.74 ± 0.29 | 44.49 ± 0.30       | 63.93 ± 0.27     | 76.10 ± 1.12   |
| **STABLE**     | 71.84 ± 0.56    | 51.36 ± 1.24 | 48.36 ± 0.21       | 41.00 ± 1.18     | 80.46 ± 2.44   |

## How to Contribute
As an active research topic, we are witenessing the rapid development of GSL methods.
Hence, this project will be frequently updated and we welcome everyone interested in this topic to contribute! 

Please feel free to send PR or issue!

## Citation
Our paper on this benchmark will be released soon!

If you use our benchmark in your works, we would appreciate citations to the paper:

```bibtex
@article{zhou2023opengsl,
  title={OpenGSL: A Comprehensive Benchmark for Graph Structure Learning},
  author={Zhiyao Zhou, Sheng Zhou, Bochao Mao, Xuanyi Zhou, Jiawei Chen, Qiaoyu Tan, Daochen Zha, Can Wang, Yan Feng, Chun Chen},
  journal={arXiv preprint},
  year={2023}
}
```

## Reference

| **Paper** | **Method** | **Conference** |
|---------|:----------:|:--------------:|
| Semi-supervised classification with graph convolutional networks      | GCN        | ICLR 2017 |
| Graph Structure Learning for Robust Graph Neural Networks  | ProGNN     | KDD 2020 |
| Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings  | IDGL       | NeurIPS 2020 |
| Graph-Revised Convolutional Network  | GRCN       | ECML-PKDD 2020 |
| Data Augmentation for Graph Neural Networks  | GAug(O)    | AAAI 2021 |
| SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks  | SLAPS      | ICML 2021 |
| Masked label prediction: Unified message passing model for semi-supervised classification  | GT         | IJCAI 2021 |
| Nodeformer: A scalable graph structure learning transformer for node classification  | Nodeformer | NeurIPS 2022 |
| Graph Structure Estimation Neural Networks  | GEN        | WWW 2021 |
| Compact Graph Structure Learning via Mutual Information Compression  | CoGSL      | WWW 2022 |
| SE-GSL: A General and Effective Graph Structure Learning Framework through Structural Entropy Optimization  | SEGSL      | WWW 2023 |
| Towards Unsupervised Deep Graph Structure Learning  | SUBLIME    | WWW 2022 |
| Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN  | STABLE     | KDD 2022 |