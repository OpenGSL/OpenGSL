# Hyperparamter Tuning

This file folder contains the hyperparameter settings for all implemented methods across all datasets. You can specify the configuration file you wish to use in the scripts (OpenGSL/paper/scripts). We rigorously followed the hyperparameter settings of papers or source codes. For those that do not provide hyperparameter settings, we performed tuning using bayesian search. 

Below are the search spaces of hyperparameters for each method. Note that for GSL methods that use GCN as a backbone, we set *n_layers*, *n_hidden* and *dropout* as GCN to reduce consumption. For GEN and SEGSL, we also set *lr, weight_decay* the same as GCN.

The hyperparameter tuning tool has not yet been integrated into OpenGSL, which will be resolved soon.

### GCN:

n_layers:[2, 3, 4, 5] 

n_hidden: [16,32,64,128]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

dropout: [0, 0.2, 0.5, 0.8]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

### GRCN:

K: [5, 50, 100, 200]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

lr_graph: [1e-1, 1e-2, 1e-3, 1e-4]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

### SLAPS:

lr: [1e-1, 1e-2, 1e-3, 1e-4]

lr_dae: [1e-2, 1e-3]

dropout_adj1: [0.25, 0.5]

dropout_adj2: [0.25, 0.5]

k: [10, 15, 20, 30]

lambda: [0.1, 1, 10, 100, 500]

ratio: [1,5,10]

nr: [1,5]

### GT:

n_layers: [2, 3, 4, 5] 

n_hidden: [16,32,64,128]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

dropout: [0, 0.2, 0.5, 0.8]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

n_heads: [1, 2, 4, 8]

### Nodeformer:

n_layers: [2, 3, 4, 5] 

n_hidden: [16,32,64,128]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

dropout: [0, 0.2, 0.5, 0.8]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

n_heads: [1, 2, 4, 8]

K: [5, 10, 20]

lambda: [1, 0.1, 0.01]

### GEN:

k: [5, 6, 7, 8, 9, 10]

tolerance: [1e-2, 1e-3, 1e-4]

threshold: [0.5, 0.6, 0.7, 0.8]

### SEGSL:

K: [2, 3, 4, 5, 6]

se: [2, 3, 4, 5, 6]

### ProGNN:

lr: [1e-1, 1e-2, 1e-3, 1e-4]

lr_adj: [1e-1, 1e-2, 1e-3, 1e-4]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

### GAug

alpha: [0, 0.2, 0.4, 0.6, 0.8, 1]

temperature: [0.3, 0.6, 0.9, 1.2]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

warm_up : [0, 10, 20]

### IDGL:

num_anchors: [300, 500, 700]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

graph_learn_num_pers: [2, 4, 6, 8]

graph_skip_conn: [0.7, 0.8, 0.9]

update_adj_ratio: [0.1, 0.2, 0.3]

### SUBLIME:

dropout rate for edge: [0, 0.25, 0.5]

$\tau$ for bootstrapping: [0.99, 0.999, 0.9999]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]

### STABLE:

threshold for consine similarity: [0.1, 0.2, 0.3]

threshold for jaccard: [0, 0.01, 0.02, 0.03]

k: [1, 3, 5, 7]

lr: [1e-1, 1e-2, 1e-3, 1e-4]

weight_decay: [5e-4. 5e-5, 5e-6, 5e-7, 0]



