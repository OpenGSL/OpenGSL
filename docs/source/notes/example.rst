:github_url: https://github.com/OpenGSL/OpenGSL

A Simple Example
========================

Following example shows you how to perform GCN on Cora dataset.

Step 1: Load configuration
----------------------------------

.. code-block:: python

    import opengsl
    conf = opengsl.config.load_conf(method="gcn", dataset="cora")

Step 2: Load data
---------------------------------------------

.. code-block:: python

    dataset = opengsl.data.Dataset("cora", n_splits=1, feat_norm=conf.dataset['feat_norm'])

Step 3: Build Model
--------------------------------

.. code-block:: python

    solver = opengsl.method.GCNSolver(conf,dataset)

Step 4: Training and Evaluation
---------------------------------

.. code-block:: python

    exp = opengsl.ExpManager(solver)
    exp.run(n_runs = 10)

