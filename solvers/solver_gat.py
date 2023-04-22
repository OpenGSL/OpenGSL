from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import torch
from models.gat import GAT
import time
from utils.utils import accuracy, AverageMeter, sample_mask, set_seed
from collections import Iterable
from .solver import BaseSolver

# split_seeds = [0,1,2,3,4]
# train_seeds = [0,1,2,3,4,5,6,7,8,9,
#                10,11,12,13,14,15,16,17,18,19,
#                20,21,22,23,24,25,26,27,28,29,
#                30,31,32,33,34,35,36,37,38,39,
#                40,41,42,43,44,45,46,47,48,49]
split_seeds = [i for i in range(20)]
train_seeds = [i for i in range(400)]


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("GAT"))
        if isinstance(self.conf.num_heads, Iterable):
            self.heads = self.conf.num_heads + [self.conf.num_out_heads]
        else:
            self.heads = [self.conf.num_heads] * (self.conf.n_layers-1) + [self.conf.num_out_heads]
        self.g = dgl.add_self_loop(self.g)

    def train(self):
        if self.conf.batch_size <= 0:
            result = self.train_non_batch()
            return result

        self.reset()
        self.start_time = time.time()
        losses = AverageMeter()

        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            losses.reset()

            for step, (input_nodes, seeds, blocks) in enumerate(self.dataloader):
                # Load the input features as well as output labels
                batch_inputs = self.feats[input_nodes].to(self.device)
                batch_labels = self.labels[seeds].to(self.device)
                blocks = [block.int().to(self.device) for block in blocks]

                # Compute loss and prediction
                batch_pred = self.model(blocks, batch_inputs)
                loss_train = self.loss_fn(batch_pred, batch_labels)
                self.optim.zero_grad()
                loss_train.backward()
                losses.update(loss_train.item())

                self.optim.step()

            if self.use_scheduler:
                self.scheduler.step()
                print("Current Learning Rate is {}".format(self.optim.param_groups[0]['lr']))

            loss_val, acc_val = self.evaluate(self.val_mask)

            if loss_val < self.best_val_loss:
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.weights = deepcopy(self.model.state_dict())
                self.total_time = time.time()-self.start_time

            # print
            if self.args.debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, losses.avg, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result
    
    def train_non_batch(self):
        self.reset()
        self.start_time = time.time()
        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()

            # train
            self.model.train()
            pred = self.model(self.g, self.feats)
            loss_train = self.loss_fn(pred[self.train_mask], self.labels[self.train_mask])

            self.optim.zero_grad()
            loss_train.backward()
            self.optim.step()

            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), pred[self.train_mask].detach().cpu().numpy())

            # eval
            loss_val, acc_val = self.evaluate(self.val_mask)

            if loss_val < self.best_val_loss:
                improve = '*'
                self.best_val_loss = loss_val
                self.weights = deepcopy(self.model.state_dict())
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.total_time = time.time()-self.start_time

            # print
            if self.args.debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            if self.conf.batch_size < 0:
                logits = self.model(self.g, self.feats).cpu()
            else:
                logits = self.model.inference(self.g, self.feats, self.device, self.conf.batch_size, self.conf.num_workers)
        self.model.train()
        logits = logits[test_mask]
        labels = self.labels[test_mask].cpu()
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)

    def reset(self):
        self.model = GAT(g=self.g, num_layers=self.conf.n_layers - 1, in_dim=self.dim_feats, num_hidden=self.conf.n_hidden,
                    num_classes=self.num_targets, heads=self.heads, activation=F.elu, feat_drop=self.conf.feat_drop,
                    attn_drop=self.conf.attn_drop, negative_slope=0.2, residual=self.conf.residual).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        self.use_scheduler = False
        if "scheduler" in self.conf:
            self.use_scheduler = True
            if self.conf.scheduler == "MultiStep":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=self.optim,
                    milestones=self.conf.milestones,
                    gamma=self.conf.gamma
                )
            else:
                raise ValueError("Scheduler Type {} Has Not Been Supported.".format(self.conf.scheduler))
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}
        if self.conf.batch_size > 0:
            self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.conf.n_layers)
            self.dataloader = dgl.dataloading.NodeDataLoader(
                graph=self.g,
                indices=self.train_mask,
                graph_sampler=self.sampler,
                device='cpu',
                batch_size=self.conf.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.conf.num_workers
            )
