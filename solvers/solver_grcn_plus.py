import torch.nn.functional as F
from copy import deepcopy
from models.grcn_plus import GRCN
import torch
import time
from utils.utils import accuracy
from .solver import BaseSolver
import dgl


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("grcn"))
        if self.args.data_load == 'pyg':
            loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
            edges = torch.cat([self.g.edge_index, loop_edge_index], dim=1)
            self.adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [self.n_nodes, self.n_nodes]).to(
                self.device).coalesce()
        else:
            self.g = dgl.add_self_loop(self.g)
            self.adj = self.g.adj().to(self.device).coalesce()
        tmp = torch.sparse.mm(self.adj,self.adj)
        edges_hd = tmp.coalesce().indices()
        self.adj_hd = torch.sparse.FloatTensor(edges_hd, torch.ones(edges_hd.shape[1]).to(self.device), [self.n_nodes, self.n_nodes]).to(
            self.device).coalesce()
        self.adj_ned = self.adj if self.conf.ned == 1 else self.adj_hd


    def train(self):
        self.reset()
        self.start_time = time.time()

        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim1.zero_grad()
            self.optim2.zero_grad()

            # forward and backward
            output = self.model(self.feats, self.adj, self.adj_ned)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim1.step()
            self.optim2.step()

            # Evaluate
            loss_val, acc_val, output = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())

            # print
            if self.args.debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        if self.conf.save_graph:
            torch.save(self.best_graph.cpu(), self.graph_loc)
        return self.result

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.feats, self.adj, self.adj_ned)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), output

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)

    def reset(self):
        # 这里使用reset的方式，否则train_gcn等函数需要大量参数
        self.model = GRCN(self.n_nodes, self.feats.shape[1], self.n_classes, self.device, self.conf)
        self.model = self.model.to(self.device)
        self.optim1 = torch.optim.Adam(self.model.base_parameters(), lr=self.conf.lr, weight_decay=self.conf.wd)
        self.optim2 = torch.optim.Adam(self.model.graph_parameters(), lr=self.conf.lr_graph)

        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}


