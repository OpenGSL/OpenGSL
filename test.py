import nni
if nni.get_trial_id()=="STANDALONE":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
import opengsl

data_name = ['cora', 'citeseer', 'pubmed','blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire', 'questions', 'minesweeper', 'wiki-cooc']
dataset = data_name[4]
# conf = opengsl.config.load_conf(path='./examples/configs/gin.yaml')
conf = opengsl.config.load_conf(method='gin', dataset=dataset)
dataset = opengsl.data.Dataset(dataset, n_splits=1, feat_norm=conf.dataset['feat_norm'])
solver = opengsl.method.GINSolver(conf,dataset)
exp = opengsl.ExpManager(solver)
exp.run(n_runs = 1)