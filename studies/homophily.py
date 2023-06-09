import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import torch
from utils.utils import get_homophily
from data.dataset import Dataset

fill = None
name = 'amazon-ratings'
method = 'gen'
data = Dataset(name, path='./data/')
print(name, method)
h = []
print(get_homophily(data.labels.cpu(), data.adj.to_dense().cpu(), type='edge', fill=fill))
for i in range(10):
    adj = torch.load(os.path.join('results/graph/{}'.format(method), '{}_{}_{}.pth'.format(name, 0, i)))
    h.append(get_homophily(data.labels.cpu(), adj.cpu(), type='edge', fill=fill))
    print(h)
h = np.array(h)
print(f'{h.mean():.4f} ± {h.std():.4f}')

# /root/anaconda3/envs/GSL/bin/python /mbc/GSL/GSL-Benchmark/studies/homophily.py