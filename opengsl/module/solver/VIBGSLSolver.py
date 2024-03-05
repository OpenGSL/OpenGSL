from copy import deepcopy
import time
import math
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from opengsl.module.solver import Solver
from opengsl.module.model.vibgsl import VIBGSL


class VIBGSLSolver(Solver):

    def __init__(self, conf, dataset):
        super(VIBGSLSolver, self).__init__(conf, dataset)

    def set_method(self):
        self.model = VIBGSL(self.dim_feats, num_classes=self.n_classes, **self.conf.model).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])

    def learn_gc(self, debug=False):
        '''
        This is the common learning procedure for graph classification, which is overwritten for special learning procedure.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        graph : torch.tensor
            The learned structure. `None` for GNN methods.
        '''
        if isinstance(self.dataset.data_raw, Dataset):
            train_dataset = self.dataset.data_raw[self.train_mask]
            test_dataset = self.dataset.data_raw[self.test_mask]
            val_dataset = self.dataset.data_raw[self.val_mask]
        elif isinstance(self.dataset.data_raw, list):
            train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
            test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
            val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]

        train_loader = DataLoader(train_dataset, self.conf.training['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, self.conf.training['test_batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, self.conf.training['test_batch_size'], shuffle=False)
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            loss_train = 0

            # forward and backward
            preds = []
            ground_truth = []
            self.model.train()
            for data in train_loader:
                self.optim.zero_grad()
                data = data.to(self.device)
                (mu, std), out, _, _ = self.model(data)
                loss = self.loss_fn(out, data.y.view(-1)).div(math.log(2))
                KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
                loss = loss + self.conf.training['beta'] * KL_loss
                loss.backward()
                self.optim.step()
                loss_train += loss.item() * data.num_graphs
                pred = F.softmax(out, dim=1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_train = loss_train / len(train_loader.dataset)
            acc_train = self.metric(ground_truth, preds)

            if (epoch+1) % self.conf.training['lr_decay_step_size'] == 0:
                for param_group in self.optim.param_groups:
                    param_group['lr'] = self.conf.training['lr_decay_factor'] * param_group['lr']

            # Evaluate
            preds = []
            ground_truth = []
            self.model.eval()
            loss_val = 0
            for data in val_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    _, out, _, _ = self.model(data)
                    pred = F.softmax(out, dim=1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(data.y.detach().cpu().unsqueeze(1))
                loss_val += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_val = loss_val / len(val_loader.dataset)
            acc_val = self.metric(ground_truth, preds)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            # save
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train, acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        # test
        preds = []
        ground_truth = []
        self.model.load_state_dict(self.weights)
        self.model.eval()
        loss_test = 0
        for data in test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                _, out, _, _ = self.model(data)
                pred = F.softmax(out, dim=1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
            loss_test += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
        preds = torch.vstack(preds).squeeze().numpy()
        ground_truth = torch.vstack(ground_truth).squeeze().numpy()
        loss_test = loss_test / len(test_loader.dataset)
        acc_test = self.metric(ground_truth, preds)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))
        return self.result, None

    def generate_graph(self, graphs):
        new_graphs_list = []
        for graph in graphs:
            x, edge_index, y = graph.x, graph.edge_index, graph.y
            x = x.to(self.device)
            with torch.no_grad():
                _, new_adj = self.model.learn_graph(node_features=x, graph_include_self=self.model.self_loop)
            new_edge_index, new_edge_attr = dense_to_sparse(new_adj)

            new_graph = Data(x=x, edge_index=new_edge_index, edge_attr=new_edge_attr, y=y)
            new_graphs_list.append(new_graph)
        return new_graphs_list