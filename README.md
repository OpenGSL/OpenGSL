# OpenGSL
OpenGSL is a comprehensive benchmark for Graph Structure Learning

## Overview of the benchmark
OpenGSL is a easy interface for Graph Structure Learning.


## Installation
Note: OpenGSL requires Python 3.8+
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

### Step 1: Load configuration
``` python
import opengsl
conf = opengsl.load_conf(method="gcn", dataset="cora")
```

### Step 2: Load data
``` python
dataset = opengsl.data.Dataset(data, n_splits=1, feat_norm=conf.dataset['feat_norm'])
```

### Step 3: Builde Model
``` python
method = opengsl.method.gcn(conf,dataset)
```

### Step 4: Traning and Evaluation
``` python
exp = opengsl.ExpManager(method, n_runs = 1)
exp.run()
```

## Node Classification Results
The table of node classification results here


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


