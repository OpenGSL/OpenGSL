import torch
from data.pyg_load import pyg_load_dataset
from data.hetero_load import hetero_load
from data.split import get_split
from utils.utils import normalize_feats
import numpy as np
from data.homophily_control import get_new_adj
import pickle


class Dataset:

    def __init__(self, data, feat_norm=False, verbose=True, n_splits=1, cora_split=False, homophily_control=None):
        '''
        This class loads, preprocessed and splits data. The results are saved as "self.feats, self.adj, self.labels, self.train_masks, self.val_masks, self.test_masks".
        Noth that self.adj is undirected and has no self loops.

        Parameters
        ----------
        data : the name of dataset
        feat_norm : whether to normalize the features
        verbose : whether to print statistics
        n_splits : number of data splits
        cora_split : whether adopt random splits for cora, citeseer, pubmed
        '''
        self.name = data
        self.device = torch.device('cuda')
        self.prepare_data(data, feat_norm, verbose)
        self.split_data(n_splits, cora_split, verbose)
        if homophily_control:
            self.adj = get_new_adj(self.adj, self.labels.cpu().numpy(), homophily_control)

    def prepare_data(self, ds_name, feat_norm=False, verbose=True):
        '''
        Parameters
        Load data. Homophilious data are loaded via pyg, while heterophilous data are loaded with "hetero_load".

        ----------
        ds_name : the name of dataset
        feat_norm : whether to normalize the features
        verbose : whether to print statistics

        Returns
        -------

        '''
        if ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph', 'blogcatalog',
                       'flickr']:
            self.data_raw = pyg_load_dataset(ds_name)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # unnormalized
            if ds_name == 'flickr':
                self.feats = self.feats.to_dense()
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse.FloatTensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                                [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges
            self.n_classes = self.data_raw.num_classes

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            self.adj = self.adj.to(self.device)
            # normalize features
            if feat_norm:
                self.feats = normalize_feats(self.feats)

        elif ds_name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc']:
            self.feats, self.adj, self.labels, self.splits = hetero_load(ds_name)

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            self.adj = self.adj.to(self.device)
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.n_edges = len(self.adj.coalesce().values())/2
            if feat_norm:
                self.feats = normalize_feats(self.feats)
            self.n_classes = len(self.labels.unique())

        else:
            print('dataset not implemented')
            exit(0)

        if verbose:
            print("""----Data statistics------'
                #Nodes %d
                #Edges %d
                #Classes %d""" %
                  (self.n_nodes, self.n_edges, self.n_classes))

        self.num_targets = self.n_classes
        if self.num_targets == 2:
            self.num_targets = 1

    def split_data(self, n_splits, cora_split, verbose=True):
        '''
        Parameters
        ----------
        n_splits : number of data splits
        cora_split : whether adopt random splits for cora, citeseer, pubmed
        verbose : whether to print statistics

        Returns
        -------

        '''
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        if self.name in ['blogcatalog', 'flickr']:
            def load_obj(file_name):
                with open(file_name, 'rb') as f:
                    return pickle.load(f)

            train_indices, val_indices, test_indices = load_obj('data/' + self.name + '_tvt_nids.pkl')
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
                if cora_split:
                    np.random.seed(i)
                    train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(), train_examples_per_class=20, val_size=500, test_size=1000)
                    self.train_masks.append(train_indices)
                    self.val_masks.append(val_indices)
                    self.test_masks.append(test_indices)
                else:
                    self.train_masks.append(torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy())
                    self.val_masks.append(torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy())
                    self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())
        elif self.name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc']:
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
        else:
            print('dataset not implemented')
            exit(0)

        if verbose:
            print("""----Split statistics of %d splits------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (n_splits, len(self.train_masks[0]), len(self.val_masks[0]), len(self.test_masks[0])))