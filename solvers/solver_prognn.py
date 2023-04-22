import torch.nn.functional as F
from copy import deepcopy
import torch.optim as optim
# from models.GCN3 import GCN
from models.enhancedgcn import GCN
from models.prognn import PGD, prox_operators, EstimateAdj, feature_smoothing
import torch
import time
from utils.utils import accuracy
from .solver import BaseSolver


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("prognn"))
        self.adj = self.adj.to_dense()


    def train_gcn(self, epoch):
        normalized_adj = self.estimator.normalize()

        t = time.time()
        improve = ''
        self.model.train()
        self.optimizer.zero_grad()

        # forward and backward
        output = self.model(self.feats, normalized_adj)[-1]
        loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
        acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
        loss_train.backward()
        self.optimizer.step()

        # evaluate
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)

        # save best model
        if loss_val < self.best_val_loss:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.best_graph = self.estimator.estimated_adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        if self.args.debug:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                epoch+1, time.time() -t, loss_train.item(), acc_train, loss_val, acc_val, improve))

    def train_adj(self, epoch):
        estimator = self.estimator
        t = time.time()
        improve = ''
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - self.adj, p='fro')
        normalized_adj = estimator.normalize()

        if self.conf.gsl['lambda_']:
            loss_smooth_feat = feature_smoothing(estimator.estimated_adj, self.feats)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(self.feats, normalized_adj)[-1]
        loss_gcn = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
        acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())

        #loss_symmetric = torch.norm(estimator.estimated_adj - estimator.estimated_adj.t(), p="fro")
        #loss_differential =  loss_fro + self.conf.gamma * loss_gcn + self.conf.lambda_ * loss_smooth_feat + args.phi * loss_symmetric
        loss_differential = loss_fro + self.conf.gsl['gamma'] * loss_gcn + self.conf.gsl['lambda_'] * loss_smooth_feat
        loss_differential.backward()
        self.optimizer_adj.step()
        # we finish the optimization of the differential part above, next we need to do the optimization of loss_l1 and loss_nuclear

        loss_nuclear =  0 * loss_fro
        if self.conf.gsl['beta'] != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                     + self.conf.gsl['gamma'] * loss_gcn \
                     + self.conf.gsl['alpha'] * loss_l1 \
                     + self.conf.gsl['beta'] * loss_nuclear
                     #+ self.conf.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(estimator.estimated_adj.data, min=0, max=1))

        # evaluate
        self.model.eval()
        normalized_adj = estimator.normalize()
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)

        # save the best model
        if loss_val < self.best_val_loss:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.best_graph = estimator.estimated_adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        if self.args.debug:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss(adj) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                epoch+1, time.time() - t, total_loss.item(), loss_val, acc_val, improve))

    def train(self):
        self.reset()
        self.start_time = time.time()
        for epoch in range(self.conf.training['n_epochs']):
            for i in range(int(self.conf.training['outer_steps'])):
                self.train_adj(epoch)

            for i in range(int(self.conf.training['inner_steps'])):
                self.train_gcn(epoch)
            if self.improve:
                self.wait = 0
                self.improve = False
            else:
                self.wait += 1
                if self.wait == self.conf.training['patience']:
                    print('Early stop!')
                    break

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, test_mask, normalized_adj):
        self.model.eval()
        self.estimator.eval()
        with torch.no_grad():
            logits = self.model(self.feats, normalized_adj)[-1]
        logits = logits[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        self.model.load_state_dict(self.weights)
        self.estimator.estimated_adj.data.copy_(self.best_graph)
        normalized_adj = self.estimator.normalize()
        return self.evaluate(self.test_mask, normalized_adj)

    def reset(self):
        self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'], self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'], self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'], self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        self.estimator = EstimateAdj(self.adj, symmetric=self.conf.gsl['symmetric'], device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.optimizer_adj = optim.SGD(self.estimator.parameters(), momentum=0.9, lr=self.conf.training['lr_adj'])
        self.optimizer_l1 = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_l1], lr=self.conf.training['lr_adj'], alphas=[self.conf.gsl['alpha']])
        self.optimizer_nuclear = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_nuclear],
                                     lr=self.conf.training['lr_adj'], alphas=[self.conf.gsl['beta']])
        self.wait = 0
        self.improve = False
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}


