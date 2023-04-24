import os.path

import torch
from models.gt import GT
from utils.utils import normalize_sp_tensor, accuracy, set_seed, sample_mask
from copy import deepcopy
import time
from .solver import BaseSolver
import wandb
import dgl


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("gt"))
        # prepare dgl graph
        edges = self.adj.coalesce().indices().cpu()
        self.graph = dgl.graph((edges[0], edges[1]), num_nodes=self.n_nodes, idtype=torch.int)
        # print(self.graph.adj())
        # self.graph = dgl.to_bidirected(self.graph)
        self.graph = dgl.add_self_loop(self.graph).to(self.device)

    def train(self):
        if 'analysis' in self.conf and self.conf.analysis['flag']:
            wandb.init(config=self.cfg,
                       project=self.cfg.analysis['project'],
                       dir=os.path.join('/home/zzy/NeDGSL/wandb',self.conf.analysis['dir']),
                       group=self.cfg.analysis['group'])
            wandb.define_metric("loss_val", summary="min")
            wandb.define_metric("acc_val", summary="max")
            wandb.define_metric("loss_train", summary="min")
            wandb.define_metric("acc_train", summary="max")
        model = GT(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                   self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm_type'],
                   self.conf.model['n_heads'], self.conf.model['act'], ff=self.conf.model['ff'], hidden_dim_multiplier=self.conf.model['hidden_dim_multiplier']).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        total_time = 0
        best_val_loss = 10
        weights = None
        result = {'train': 0, 'valid': 0, 'test': 0}
        start_time = time.time()
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            model.train()
            optim.zero_grad()

            # forward and backward
            x, output, _ = model(self.feats, self.graph, self.labels.cpu().numpy())

            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            optim.step()

            # Evaluate
            loss_val, acc_val, _ = self.evaluate(model, self.val_mask)
            loss_test, acc_test, _ = self.evaluate(model, self.test_mask)

            # save
            if acc_val > result['valid']:
                improve = '*'
                weights = deepcopy(model.state_dict())
                total_time = time.time() - start_time
                best_val_loss = loss_val
                result['valid'] = acc_val
                result['train'] = acc_train

            # print
            if 'analysis' in self.conf and self.conf.analysis['flag']:
                wandb.log({'epoch':epoch+1,
                           'acc_val':acc_val,
                           'loss_val':loss_val,
                           'acc_train': acc_train,
                           'loss_train': loss_train})

            if self.args.debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(total_time))
        loss_test, acc_test, homo_heads = self.test(model, weights)
        result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        if 'analysis' in self.conf and self.conf.analysis['flag']:
            wandb.log({'acc_test':acc_test})
            if self.conf.analysis['graph_analysis']:
                homo = []
                for i in range(len(homo_heads)):
                    for h in homo_heads[i]:
                        homo.append([i,h])
                wandb.log({'homo':wandb.plot.scatter(wandb.Table(data=homo, columns=['layer', 'homophily']), 'layer', 'homophily')})
            wandb.finish()
        return result

    def evaluate(self, model, test_mask, graph_analysis=False):
        model.eval()
        with torch.no_grad():
            x, output, homo_heads = model(self.feats, self.graph, self.labels.cpu().numpy(), graph_analysis)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), homo_heads

    def test(self, model, weights):
        model.load_state_dict(weights)
        return self.evaluate(model, self.test_mask, graph_analysis=self.conf.analysis['graph_analysis'])






