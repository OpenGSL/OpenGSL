import opengsl

conf = opengsl.load_conf(method='grcn', dataset="cora")
dataset = opengsl.data.Dataset('cora', n_splits=1, feat_norm=conf.dataset['feat_norm'])

method = opengsl.method.gcn(conf,dataset)
exp = opengsl.ExpManager(method, n_runs = 1)
exp.run()