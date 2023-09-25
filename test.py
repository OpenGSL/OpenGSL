import nni
if nni.get_trial_id()=="STANDALONE":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
import opengsl

data_name = ['cora', 'citeseer', 'pubmed','blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire', 'questions', 'minesweeper', 'wiki-cooc']
dataset = data_name[0]
conf = opengsl.config.load_conf(path='./opengsl/config/sublime/sublime_cora_appnp.yaml')
# conf = opengsl.config.load_conf(method='gin', dataset=dataset)
dataset = opengsl.data.Dataset(dataset, n_splits=1, feat_norm=conf.dataset['feat_norm'])
solver = opengsl.method.SUBLIMESolver(conf,dataset)
exp = opengsl.ExpManager(solver)
if nni.get_trial_id()=="STANDALONE":
    exp.run(n_runs = 10, debug=True)
else:
    exp.run(n_runs = 1)