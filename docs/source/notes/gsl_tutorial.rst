:github_url: https://github.com/OpenGSL/OpenGSL

Build Your Own GSL
========================

Let’s try to implement a simple GSL algorithm using OpenGSL.

We provide multiple choices for each component in OpenGSL.

All of them can be freely chosen and assembled into a GraphLearner as shown below.

Following shows you how to perform GCN on Cora dataset. This page can be run in `Jupyter Notebook <https://github.com/OpenGSL/OpenGSL/blob/main/examples/Build%20Your%20Own%20GSL.ipynb>`_.

Step 1: Load Data
----------------------------------

.. code-block:: python

    from opengsl.data.dataset import Dataset
    dataset = Dataset("cora", n_splits=1)
    train_mask = dataset.train_masks[0]
    val_mask = dataset.val_masks[0]
    test_mask = dataset.test_masks[0]

Step 2: Build Model
----------------------------------
It's easy to implement a simple GSL algorithm using our provided components.

Let's **choose the basic components and build the graphlearner**.

We use the **GCNDiagEncoder** as in "Graph-Revised Convolutional Network", followed by **Cosine** (metric), **KNN** (transform), **Interpolate** (fuse). Then a **GraphLearner** is built with these components.



.. code-block:: python

    import torch
    from opengsl.module.encoder import GCNEncoder, GCNDiagEncoder
    from opengsl.module import GraphLearner
    from opengsl.module.transform import KNN
    from opengsl.module.metric import Cosine
    from opengsl.module.fuse import Interpolate
    from opengsl.utils import set_seed

    device = torch.device('cuda')
    set_seed(42)
    encoder = GCNDiagEncoder(2, dataset.dim_feats)
    metric = Cosine()
    postprocess = [KNN(150)]
    fuse = Interpolate(1, 1)
    # build the graphlearner
    graphlearner = GraphLearner(encoder=encoder, metric=metric, postprocess=postprocess, fuse=fuse).to(device)
    # define gnn model
    gnn = GCNEncoder(dataset.dim_feats, nhid=64, nclass=dataset.n_classes, n_layers=2, dropout=0.5).to(device)
Step 3: Training and Evaluation Using Our Pipeline
--------------------------------
We recommend you to use our provided pipline with **GSLSolver** and **ExpManager** to simplify the process of training and evaluation. Only *set_method* needs to be customized in this pipeline.

.. code-block:: python

    from opengsl.module.solver import GSLSolver
    from opengsl import ExpManager
    import argparse
    class MyGSL(GSLSolver):
        def set_method(self):
            encoder = GCNDiagEncoder(2, dataset.dim_feats)
            metric = Cosine()
            postprocess = [KNN(150)]
            fuse = Interpolate(1, 1)
            # build the graphlearner
            self.graphlearner = GraphLearner(encoder=encoder, metric=metric, postprocess=postprocess, fuse=fuse).to(device)
            # define gnn model
            self.model = GCNEncoder(dataset.dim_feats, nhid=64, nclass=dataset.n_classes, n_layers=2, dropout=0.5).to(device)
            self.optim = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.graphlearner.parameters()}], lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])

    conf = {'model': {'n_hidden': 64, 'n_layer': 2},
        'training': {'lr': 1e-2,
        'weight_decay': 5e-4,
        'n_epochs': 100,
        'patience': None,
        'criterion': 'metric'},
        'dataset': {'feat_norm': False, 'sparse': True},
        'analysis': {'flag': False, 'save_graph': False}}
    mygsl = MyGSL(argparse.Namespace(**conf), dataset)
    exp = ExpManager(solver=mygsl)
    exp.run(n_runs=3, debug=True)
