from copy import deepcopy
from opengsl.module.model.pastel import PASTEL
import torch
import numpy as np
import time
from opengsl.module.solver import Solver
from opengsl.module.functional import normalize
import networkx as nx
import multiprocessing as mp
import math


class PASTELSolver(Solver):
    '''
    A solver to train, evaluate, test PASTEL in a run.

    Parameters
    ----------
    conf : argparse.Namespace
        Config file.
    dataset : opengsl.data.Dataset
        The dataset.

    Attributes
    ----------
    method_name : str
        The name of the method.

    Examples
    --------
    >>> # load dataset
    >>> import opengsl.dataset
    >>> dataset = opengsl.data.Dataset('cora', feat_norm=True)
    >>> # load config file
    >>> import opengsl.config.load_conf
    >>> conf = opengsl.config.load_conf('pastel', 'cora')
    >>>
    >>> solver = PASTELSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''

    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "pastel"
        print("Solver Version : [{}]".format("pastel"))
        
        self.conf = conf
        self.adj = normalize(self.adj)


    def learn_nc(self, debug=False):
        '''
        Learning process of PASTEL.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        graph : torch.tensor
            The learned structure.
        '''
        
        for epoch in range(1, self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            
            # Calculate the shortest path dists every n epochs
            if epoch % self.conf.training['pe_every_epochs'] == 0:
                self.model.position_flag = 1
                self.shortest_path_dists = self.cal_shortest_path_distance(self.cur_adj, 5)
                self.shortest_path_dists_anchor = torch.from_numpy(self.cal_spd(self.cur_adj, 0)).to(self.device).to(torch.float32)
                self.model.position_encoding = self.shortest_path_dists_anchor
            else:
                self.model.position_flag = 0

            # forward and backward
            output, raw_adj, self.cur_adj = self.model(self.feats, self.adj)
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            loss += self.add_graph_loss(raw_adj, self.feats)
            loss.backward()
            self.optim.step()
            
            # Calculate group pagerank
            if epoch % self.conf.training['gpr_every_epochs'] == 0:
                self.group_pagerank_after = self.cal_group_pagerank(self.cur_adj, 0.85)
                self.group_pagerank_args = torch.from_numpy(self.cal_group_pagerank_args(self.group_pagerank_before, self.group_pagerank_after)).to(self.device).to(torch.float32)
                self.model.gpr_rank = self.group_pagerank_args

            # Evaluate
            loss_val, acc_val, adjs = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
                if self.conf.analysis['save_graph']:
                    self.adjs['new'] = adjs['new'].to_dense().detach().clone()
                    self.adjs['final'] = adjs['final'].to_dense().detach().clone()

            # print

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _= self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.adjs

    def evaluate(self, test_mask):
        '''
        Evaluation procedure of PASTEL.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        adj : torch.tensor
            The learned structure.
        '''
        self.model.eval()
        with torch.no_grad():
            output, adjs, _ = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), adjs

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.cur_adj = self.adj
        self.anchor_sets = self.select_anchor_sets()
        
        # Calculate the shortest path dists
        self.shortest_path_dists = np.zeros((self.n_nodes, self.n_nodes))
        self.shortest_path_dists = self.cal_shortest_path_distance(self.cur_adj, 5)

        self.shortest_path_dists_anchor = np.zeros((self.n_nodes, self.n_nodes))
        self.shortest_path_dists_anchor = torch.from_numpy(self.cal_spd(self.cur_adj, 0)).to(self.device).to(torch.float32)

        # Calculate group pagerank
        self.group_pagerank_before = self.cal_group_pagerank(self.cur_adj, 0.85)
        self.group_pagerank_after = self.group_pagerank_before
        self.group_pagerank_args = torch.from_numpy(self.cal_group_pagerank_args(self.group_pagerank_before, self.group_pagerank_after)).to(self.device).to(torch.float32)

        # Calculate avg spd before training:
        self.labeled_idx = np.array(self.train_mask)
        self.unlabeled_idx = np.append(self.val_mask, self.test_mask)
        
        self.model = PASTEL(self.n_nodes, self.dim_feats, self.num_targets, self.conf, self.shortest_path_dists_anchor, self.group_pagerank_args).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                       weight_decay=self.conf.training['weight_decay'])
        
        
    def add_graph_loss(self, out_adj, features):
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.conf.training['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = torch.ones(out_adj.size(-1)).cuda()
        graph_loss += -self.conf.training['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
        graph_loss += self.conf.training['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss
    
    
    def select_anchor_sets(self):
        num_classes = self.n_classes
        n_anchors = 0

        class_anchor_num = [0 for _ in range(num_classes)]
        anchor_nodes = [[] for _ in range(num_classes)]
        anchor_node_list = []
    
        idx_train = self.train_mask
        labels = self.labels

        for iter1 in idx_train:
            iter_label_index = labels[iter1]
            anchor_nodes[iter_label_index].append(iter1)
            class_anchor_num[iter_label_index] += 1
            anchor_node_list.append(iter1)
            n_anchors += 1

        self.num_anchors = n_anchors
        self.anchor_node_list = anchor_node_list
        self.conf.model['num_anchors'] = self.num_anchors
        return anchor_nodes
    
    
    def cal_spd(self, adj, approximate):
        num_anchors = self.num_anchors
        num_nodes = self.n_nodes
        spd_mat = np.zeros((num_nodes, num_anchors))
        shortest_path_distance_mat = self.shortest_path_dists
        for iter1 in range(num_nodes):
            for iter2 in range(num_anchors):
                spd_mat[iter1][iter2] = shortest_path_distance_mat[iter1][self.anchor_node_list[iter2]]

        max_spd = np.max(spd_mat)
        spd_mat = spd_mat / max_spd

        return spd_mat
    
    
    def cal_shortest_path_distance(self, adj, approximate):
        n_nodes = self.n_nodes
        Adj = adj.to_dense().detach().cpu().numpy()
        G = nx.from_numpy_array(Adj)
        G.edges(data=True)
        dists_array = np.zeros((n_nodes, n_nodes))
        dists_dict = all_pairs_shortest_path_length_parallel(G, cutoff=approximate if approximate > 0 else None)

        cnt_disconnected = 0

        for i, node_i in enumerate(G.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(G.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist == -1:
                    cnt_disconnected += 1
                if dist != -1:
                    dists_array[node_i, node_j] = dist
        return dists_array


    def rank_group_pagerank(self, pagerank_before, pagerank_after):
        pagerank_dist = torch.mm(pagerank_before, pagerank_after.transpose(-1, -2)).detach().cpu()
        num_nodes = self.n_nodes
        node_pair_group_pagerank_mat = np.zeros((num_nodes, num_nodes))
        node_pair_group_pagerank_mat_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat_list.append(pagerank_dist[i, j])
        node_pair_group_pagerank_mat_list = np.array(node_pair_group_pagerank_mat_list)
        index = np.argsort(-node_pair_group_pagerank_mat_list)
        rank = np.argsort(index)
        rank = rank + 1
        iter = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat[i][j] = rank[iter]
                iter = iter + 1

        return node_pair_group_pagerank_mat


    def cal_group_pagerank(self, adj, pagerank_prob):
        num_nodes = self.n_nodes
        num_classes = self.n_classes

        labeled_list = [0 for _ in range(num_classes)]
        labeled_node = [[] for _ in range(num_classes)]
        labeled_node_list = []

        idx_train = self.train_mask
        labels = self.labels

        for iter1 in idx_train:
            iter_label = labels[iter1]
            labeled_node[iter_label].append(iter1)
            labeled_list[iter_label] += 1
            labeled_node_list.append(iter1)

        if (num_nodes > 5000):
            A = adj.detach()
            A_hat = A.to(self.device) + torch.eye(A.size(0)).to(self.device)
            D = torch.sum(A_hat, 1)
            D_inv = torch.eye(num_nodes).to(self.device)

            for iter in range(num_nodes):
                if (D[iter] == 0):
                    D[iter] = 1e-12
                D_inv[iter][iter] = 1.0 / D[iter]
            D = D_inv.sqrt().to(self.device)

            A_hat = torch.mm(torch.mm(D, A_hat), D)
            temp_matrix = torch.eye(A.size(0)).to(self.device) - pagerank_prob * A_hat
            temp_matrix = temp_matrix.cpu().numpy()
            temp_matrix_inv = np.linalg.inv(temp_matrix).astype(np.float32)

            inv = torch.from_numpy(temp_matrix_inv).to(self.device)
            P = (1 - pagerank_prob) * inv

        else:
            A = adj
            A_hat = torch.eye(A.size(0)).to(self.device) + A.to(self.device)
            D = torch.diag(torch.sum(A_hat, 1))
            D = D.inverse().sqrt()
            A_hat = torch.mm(torch.mm(D, A_hat), D)
            P = (1 - pagerank_prob) * ((torch.eye(A.size(0)).to(self.device) - pagerank_prob * A_hat).inverse())

        I_star = torch.zeros(num_nodes)

        for class_index in range(num_classes):
            Lc = labeled_list[class_index]
            Ic = torch.zeros(num_nodes)
            Ic[torch.tensor(labeled_node[class_index])] = 1.0 / Lc
            if class_index == 0:
                I_star = Ic
            if class_index != 0:
                I_star = torch.vstack((I_star,Ic))

        I_star = I_star.transpose(-1, -2).to(self.device)

        Z = torch.mm(P, I_star)
        return Z


    def cal_group_pagerank_args(self, pagerank_before, pagerank_after):
        node_pair_group_pagerank_mat = self.rank_group_pagerank(pagerank_before, pagerank_after) # rank
        num_nodes = self.n_nodes
        PI = 3.1415926
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat[i][j] = 2 - (math.cos((node_pair_group_pagerank_mat[i][j] / (num_nodes * num_nodes)) * PI) + 1)

        return node_pair_group_pagerank_mat


    def rank_group_pagerank_KL(self, pagerank_before, pagerank_after): # KL
        num_nodes = self.n_nodes

        KL_A = pagerank_before[0]
        KL_B = pagerank_after

        for i in range(num_nodes):
            if i == 0:
                for j in range(num_nodes-1):
                    KL_A = torch.vstack((KL_A, pagerank_before[i]))
            else:
                for j in range(num_nodes):
                    KL_A = torch.vstack((KL_A, pagerank_before[i]))

        for i in range(num_nodes-1):
            KL_B = torch.vstack((KL_B, pagerank_after))

        pagerank_dist = torch.nn.functional.kl_div(KL_A.softmax(dim=-1).log(), KL_B.softmax(dim=-1), reduction='none').detach()
        pagerank_dist = torch.sum(pagerank_dist, dim=1) * (-1)

        node_pair_group_pagerank_mat_list = pagerank_dist.flatten()
        index = torch.argsort(-node_pair_group_pagerank_mat_list)
        rank = torch.argsort(index)
        rank = rank + 1
        node_pair_group_pagerank_mat = torch.reshape(rank, ((num_nodes, num_nodes)))

        return node_pair_group_pagerank_mat
    
    
def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)   # unweighted
    return dists_dict

def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=1)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(graph, nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result