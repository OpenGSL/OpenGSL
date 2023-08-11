#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.
# Modified from https://github.com/jianhao2016/GPRGNN/blob/master/src/cSBM_dataset.py

"""
This is a script for contexual SBM model and its dataset generator.
contains functions:
        ContextualSBM
        parameterized_Lambda_and_mu
        save_data_to_pickle
    class:
        dataset_ContextualSBM

"""
import torch
import pickle
from datetime import datetime
import os.path as osp
import os
import urllib.request

import torch
from torch_geometric.data import InMemoryDataset


class dataset_ContextualSBM(InMemoryDataset):
    r"""Create synthetic dataset based on the contextual SBM from the paper:
    https://arxiv.org/pdf/1807.09596.pdf

    Use the similar class as InMemoryDataset, but not requiring the root folder.

       See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset if not specified use time stamp.

        for {n, d, p, Lambda, mu}, with '_' as prefix: intial/feed in argument.
        without '_' as prefix: loaded from data information

        n: number nodes
        d: avg degree of nodes
        p: dimenstion of feature vector.

        Lambda, mu: parameters balancing the mixture of information, 
                    if not specified, use parameterized method to generate.

        epsilon, theta: gap between boundary and chosen ellipsoid. theta is 
                        angle of between the selected parameter and x-axis.
                        choosen between [0, 1] => 0 = 0, 1 = pi/2

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

#     url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name=None,
                 n=800, d=5, p=100, Lambda=None, mu=None,
                 epsilon=0.1, theta=0.5,
                 train_percent=0.025,
                 transform=None, pre_transform=None):

        now = datetime.now()
        surfix = now.strftime('%b_%d_%Y-%H:%M')
        if name is None:
            # not specifing the dataset name, create one with time stamp.
            self.name = '_'.join(['cSBM_data', surfix])
        else:
            self.name = name

        self._n = n
        self._d = d
        self._p = p

        self._Lambda = Lambda
        self._mu = mu
        self._epsilon = epsilon
        self._theta = theta

        self._train_percent = train_percent

        root = osp.join(root, self.name)
        if not osp.isdir(root):
            os.makedirs(root)
        super(dataset_ContextualSBM, self).__init__(
            root, transform, pre_transform)

#         ipdb.set_trace()
        self.data, self.slices = torch.load(self.processed_paths[0])
        # overwrite the dataset attribute n, p, d, Lambda, mu
        self.Lambda = self.data.Lambda.item()
        self.mu = self.data.mu.item()
        self.n = self.data.n
        self.p = self.data.p
        self.d = self.data.d
        self.train_percent = self.data.train_percent

#     @property
#     def raw_dir(self):
#         return osp.join(self.root, self.name, 'raw')

#     @property
#     def processed_dir(self):
#         return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.raw_dir, name)
            if not osp.isfile(p2f):
                url = 'https://raw.githubusercontent.com/OpenGSL/HeterophilousDatasets/main/cSBM/'
                
                try:
                    print('Downloading', url+name)
                    urllib.request.urlretrieve(url + name, os.path.join(self.root, name))
                    print('Done!')
                except:
                    raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)