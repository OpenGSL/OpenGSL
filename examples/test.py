import opengsl
data_name = ['cora', 'pubmed', 'citeseer','blogcatalog', 'flickr', 'amazon-ratings', 'questions', 'minesweeper', 'roman-empire', 'wiki-cooc']
dataset = data_name[0]
conf = opengsl.config.load_conf(path='./examples/configs/gin.yaml')
dataset = opengsl.data.Dataset(dataset, n_splits=1, feat_norm=conf.dataset['feat_norm'])
solver = opengsl.method.GINSolver(conf,dataset)
exp = opengsl.ExpManager(solver)
exp.run(n_runs = 1)
