import torch
from opengsl.data.dataset.pyg_load import pyg_load_dataset
from opengsl.data.dataset.hetero_load import hetero_load
from opengsl.data.dataset.split import get_split, k_fold
from opengsl.data.preprocess.knn import knn
from opengsl.module.functional import normalize
import numpy as np
from opengsl.data.preprocess.control_homophily import control_homophily
import pickle
import os
import urllib.request
from torch_geometric.utils import degree
import torch_geometric.transforms as T
# from ogb.nodeproppred import PygNodePropPredDataset


class Dataset:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        The name of dataset.
    feat_norm : bool
        Whether to normalize the features.
    verbose : bool
        Whether to print statistics.
    n_splits : int
        Number of data splits.
    homophily_control : float
        The homophily ratio `control homophily` receives. If set to `None`, the original adj will be kept unchanged.
    path : str
        Path to save dataset files.
    '''

    def __init__(self, data, feat_norm=False, verbose=True, n_splits=1, homophily_control=None, path='./data/',
                 without_structure=None, train_percent=None, val_percent=None):
        self.name = data
        self.path = path
        self.device = torch.device('cuda')
        self.single_graph = True
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.prepare_data(data, feat_norm, verbose)
        if self.single_graph:
            self.split_data(n_splits, verbose)
        else:
            self.split_graphs(n_splits, verbose)
        if homophily_control:
            self.adj = control_homophily(self.adj, self.labels.cpu().numpy(), homophily_control)
        # zero knowledge on structure
        if without_structure:
            if without_structure == 'i':
                self.adj = torch.eye(self.n_nodes).to(self.device).to_sparse()
            elif without_structure == 'knn':
                self.adj = knn(self.feats, int(self.n_edges//self.n_nodes)).to_sparse()

    def prepare_data(self, ds_name, feat_norm=False, verbose=True):
        '''
        Function to Load various datasets.
        Homophilous datasets are loaded via pyg, while heterophilous datasets are loaded with `hetero_load`.
        The results are saved as `self.feats, self.adj, self.labels, self.train_masks, self.val_masks, self.test_masks`.
        Noth that `self.adj` is undirected and has no self loops.

        Parameters
        ----------
        ds_name : str
            The name of dataset.
        feat_norm : bool
            Whether to normalize the features.
        verbose : bool
            Whether to print statistics.

        '''
        if ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph', 'blogcatalog',
                       'flickr', 'wikics'] or 'csbm' in ds_name:
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # unnormalized
            if ds_name == 'flickr':
                self.feats = self.feats.to_dense()
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse.FloatTensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                                [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges/2
            self.n_classes = self.data_raw.num_classes

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            
            if 'csbm' in ds_name:
                self.labels = self.labels.float()
            
            self.adj = self.adj.to(self.device)
            # normalize features
            if feat_norm:
                self.feats = normalize(self.feats, style='row')

        elif ds_name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'tolokers']:
            self.feats, self.adj, self.labels, self.splits = hetero_load(ds_name, path=self.path)

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            self.adj = self.adj.to(self.device)
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.n_edges = len(self.adj.coalesce().values())/2
            if feat_norm:
                self.feats = normalize(self.feats, style='row')
                # exit(0)
            self.n_classes = len(self.labels.unique())

        # elif ds_name in ['ogbn-arxiv']:
        #     self.data_raw = PygNodePropPredDataset(name='ogbn-arxiv', root='./data')
        #     self.g = self.data_raw[0]
        #     self.feats = self.g.x  # unnormalized
        #     self.n_nodes = self.feats.shape[0]
        #     self.dim_feats = self.feats.shape[1]
        #     self.labels = self.g.y
        #     reverse_edge_index = torch.stack([self.g.edge_index[1], self.g.edge_index[0]])
        #
        #     self.adj = torch.sparse.FloatTensor(torch.cat([reverse_edge_index, self.g.edge_index], dim=1), torch.ones(self.g.edge_index.shape[1]*2),
        #                                         [self.n_nodes, self.n_nodes])
        #     self.n_edges = self.g.num_edges
        #     self.n_classes = self.data_raw.num_classes
        #
        #     self.feats = self.feats.to(self.device)
        #     self.labels = self.labels.to(self.device).view(-1)
        #     self.adj = self.adj.to(self.device)
        
        elif ds_name in ['regression']:
            def read_regression(input_folder):
                import pandas as pd
                import json
                import networkx as nx
                X = pd.read_csv(f'{input_folder}/X.csv')
                y = pd.read_csv(f'{input_folder}/y.csv')

                networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
                networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})

                categorical_columns = []
                if os.path.exists(f'{input_folder}/cat_features.txt'):
                    with open(f'{input_folder}/cat_features.txt') as f:
                        for line in f:
                            if line.strip():
                                categorical_columns.append(line.strip())

                cat_features = None
                if categorical_columns:
                    columns = X.columns
                    cat_features = np.where(columns.isin(categorical_columns))[0]

                    for col in list(columns[cat_features]):
                        X[col] = X[col].astype(str)


                if os.path.exists(f'{input_folder}/masks.json'):
                    with open(f'{input_folder}/masks.json') as f:
                        masks = json.load(f)
                else:
                    print('no inside split masks')
                
                X = torch.from_numpy(X.values).to(torch.float)
                y = torch.from_numpy(y.values).squeeze(1).to(torch.float)
                
                adj = nx.to_scipy_sparse_array(networkx_graph).tocoo()
                # 获取非零元素行索引
                row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
                # 获取非零元素列索引
                col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
                # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
                edge_index = torch.stack([row, col], dim=0)
                adj = torch.sparse_coo_tensor(edge_index, torch.ones_like(row))
                    
                return X, adj, y, masks
            
            self.feats,self.adj,self.labels,masks = read_regression('./data/regression')
            
            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            self.adj = self.adj.to(self.device)
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.n_edges = len(self.adj.coalesce().values())/2
            self.n_classes = 1
            self.masks = masks

        elif ds_name in ["IMDB-BINARY", "REDDIT-BINARY", "COLLAB", "IMDB-MULTI"]:
            # graph level
            self.single_graph = False
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            if self.data_raw.data.x is None:
                # 如果没有节点特征，使用度数作为节点特征
                # 所有图的最大度数
                max_degree = 0
                degs = []
                for data in self.data_raw:
                    degs += [degree(data.edge_index[0], dtype=torch.long)]
                    max_degree = max(max_degree, degs[-1].max().item())

                # 两种安排度数特征的方式
                if max_degree < 1000:
                    self.data_raw.transform = T.OneHotDegree(max_degree)
                else:
                    deg = torch.cat(degs, dim=0).to(torch.float)
                    mean, std = deg.mean().item(), deg.std().item()
                    self.data_raw.transform = NormalizedDegree(mean, std)
            self.n_graphs = len(self.data_raw)
            self.n_classes = self.data_raw.num_classes
            self.dim_feats = self.data_raw[0].x.shape[1]

        else:
            print('dataset not implemented')
            exit(0)

        if verbose:
            if self.single_graph:
                print("""----Data statistics------'
                    #Nodes %d
                    #Edges %d
                    #Classes %d""" %
                      (self.n_nodes, self.n_edges, self.n_classes))
            else:
                print("""----Data statistics------'
                                    #Graphs %d
                                    #Classes %d""" %
                      (self.n_graphs, self.n_classes))

        self.num_targets = self.n_classes
        if self.num_targets == 2:
            self.num_targets = 1

    def split_data(self, n_splits, verbose=True):
        '''
        Function to conduct data splitting for various datasets.

        Parameters
        ----------
        n_splits : int
            Number of data splits.
        verbose : bool
            Whether to print statistics.

        '''
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        if self.name in ['blogcatalog', 'flickr']:
            def load_obj(file_name):
                with open(file_name, 'rb') as f:
                    return pickle.load(f)
            def download(name):
                url = 'https://github.com/zhao-tong/GAug/raw/master/data/graphs/'
                try:
                    print('Downloading', url + name)
                    urllib.request.urlretrieve(url + name, os.path.join(self.path, name))
                    print('Done!')
                except:
                    raise Exception(
                        '''Download failed! Make sure you have stable Internet connection and enter the right name''')

            split_file = self.name + '_tvt_nids.pkl'
            if not os.path.exists(os.path.join(self.path, split_file)):
                download(split_file)
            train_indices, val_indices, test_indices = load_obj(os.path.join(self.path, split_file))
            for i in range(n_splits):
                self.train_masks.append(train_indices)
                self.val_masks.append(val_indices)
                self.test_masks.append(test_indices)

        elif self.name in ['coauthorcs', 'coauthorph', 'amazoncom', 'amazonpho']:
            for i in range(n_splits):
                np.random.seed(i)
                train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(), train_examples_per_class=20, val_examples_per_class=30)  # 默认采取20-30-rest这种划分
                self.train_masks.append(train_indices)
                self.val_masks.append(val_indices)
                self.test_masks.append(test_indices)
        elif self.name in ['cora', 'citeseer', 'pubmed']:
            for i in range(n_splits):
                self.train_masks.append(torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy())
                self.val_masks.append(torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy())
                self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())
        elif self.name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'tolokers']:
            assert n_splits < 10 , 'n_splits > splits provided'
            self.train_masks = self.splits[0][:n_splits]
            self.val_masks = self.splits[1][:n_splits]
            self.test_masks = self.splits[2][:n_splits]
        # elif ds_name in ['penn94']:
        #     train_indices = self.splits[seed]['train']
        #     val_indices = self.splits[seed]['valid']
        #     test_indices = self.splits[seed]['test']
        #     self.train_mask = generate_mask_tensor(sample_mask(train_indices, self.n_nodes))
        #     self.val_mask = generate_mask_tensor(sample_mask(val_indices, self.n_nodes))
        #     self.test_mask = generate_mask_tensor(sample_mask(test_indices, self.n_nodes))
        elif self.name in ['ogbn-arxiv']:
            split_idx = self.data_raw.get_idx_split()
            train_idx = split_idx['train']
            val_idx = split_idx['valid']
            test_idx = split_idx['test']
            for i in range(n_splits):
                self.train_masks.append(train_idx.numpy())
                self.val_masks.append(val_idx.numpy())
                self.test_masks.append(test_idx.numpy())
        elif self.name in ['wikics']:
            for i in range(n_splits):
                self.train_masks.append(torch.nonzero(self.g.train_mask[:,i], as_tuple=False).squeeze().numpy())
                self.val_masks.append(torch.nonzero(self.g.val_mask[:,i], as_tuple=False).squeeze().numpy())
                self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())
        elif 'csbm' in self.name:
            for i in range(n_splits):
                np.random.seed(i)
                train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(), train_size=int(self.n_nodes*self.train_percent), val_size=int(self.n_nodes*self.val_percent))
                self.train_masks.append(train_indices)
                self.val_masks.append(val_indices)
                self.test_masks.append(test_indices)
        elif self.name in ['regression']:
            for i in range(n_splits):
                self.train_masks.append(self.masks[str(i)]['train'])
                self.val_masks.append(self.masks[str(i)]['val'])
                self.test_masks.append(self.masks[str(i)]['test'])
        else:
            print('dataset not implemented')
            exit(0)

        if verbose:
            print("""----Split statistics of %d splits------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (n_splits, len(self.train_masks[0]), len(self.val_masks[0]), len(self.test_masks[0])))

    def split_graphs(self, n_splits, verbose=True):
        '''
        Function to conduct data splitting for graph-level datasets.

        Parameters
        ----------
        n_splits : int
            Number of data splits.
        verbose : bool
            Whether to print statistics.

        '''
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(self.data_raw, n_splits))):
            self.train_masks.append(train_idx)
            self.val_masks.append(val_idx)
            self.test_masks.append(test_idx)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


if __name__ == '__main__':
    Dataset('IMDB-BINARY', n_splits=10)