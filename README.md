# GSL Benchmark

> A GSL Benchmark for GSL Research

## Files in the folder

GSL-Benchmark

- configs							config files for various methods

- data

  - \_init\_.py					load data via dgl, deprecated

  - hetero_load.py        load data from *A critical look at evaluation of GNNs under heterophily* [paper](https://arxiv.org/abs/2302.11640) [code](https://github.com/heterophily-submit/HeterophilousDatasets)

  - ogb_data.py             load ogb data

  - pyg_load.py              load data via pyg, recommended

  - split.py                      functions to split data into train/val/test 

- models                             modules of various methods
  - GCN3.py                   GCN module
  - GCN2.py                   GCN module supporting different dropout for training and evaluating, specially for IDGL, expected to be integrated
  - enhancedgcn.py      GCN module supporting input layer, output layer, residual connection, layer normalization, various activation functions, ..., recommended
  - other methods...
- solvers
  - solver.py					base solver, loading data and running multiple experiments
  - solver_gcndense.py solver of gcn for training, evaluating, testing in a run
  - other solvers
- utils                                     util functions
- main.py 
- README.md                             

## Explanation

Different methods share a base solver, with the same data loading, splitting and multiple experimental processes. Due to the large differences in the training methods of different methods, we find it difficult to unify the training, evaluating, testing process. Thus Different methods have their own solver inherited from the base solver, responsible for the training, evaluating, testing in a single run.  

## Usage 

Please read main.py for more information about args to be entered.

Here are some examples:

```python
python main.py --data cora --solver gcndense --config configs/gcn/gcn_cora.yaml --gpu 0
python main.py --data cora --solver grcn --config configs/grcn/grcn_cora.yaml --gpu 0 --not_norm_feats
python main.py --data wiki-cooc --solver gt --config configs/gt/gt_hetero.yaml --gpu 0 --data_load hetero --not_norm_feats
```

## Performances

## Explorations

