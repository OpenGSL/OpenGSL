# Reproducibility
Here we provide codes to reproduce the results in our paper, which also serve as examples to use the library.

To reproduce the main results, run

`python main_results.py --data {dataset} --method {gsl method} --debug {whether print training details} --gpu {specify gpu}
`

Besides getting the performances of the specified GSL method, the learned structures will also be saved in paper/results/graph, which are needed to run the experiments below.

To reproduce the results on sec4.2, run

`python homophily.py --data {dataset} --method {gsl method}
`

To reproduce the results on sec4.3, run
`
python generalizability.py --data {dataset} --gsl {source of learned strucures} --gnn {GNN model} --debug {whether print training details} --gpu {specify gpu}
`

