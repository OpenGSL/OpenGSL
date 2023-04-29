import torch.nn.functional as F
import torch
from models.appnp import MyAPPNP as APPNP
from utils.utils import normalize_sp_tensor, accuracy, set_seed, sample_mask
from copy import deepcopy
import time
from .solver import BaseSolver


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("gcndense"))
        # self.edge_index = self.adj.coalesce().indices()
        self.normalize = normalize_sp_tensor if self.conf.dataset['normalize'] else lambda x, y: x

    def train(self):
        self.normalized_adj = self.normalize(self.adj, self.conf.dataset['add_loop'])
        model = APPNP(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, dropout=self.conf.model['dropout'], K=self.conf.model['K'], alpha=self.conf.model['alpha']).to(self.device)
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
            output = model(self.feats, self.normalized_adj)
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
            if self.args.debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(total_time))
        loss_test, acc_test, _ = self.test(model, weights)
        result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return result

    def evaluate(self, model, test_mask):
        model.eval()
        with torch.no_grad():
            output = model(self.feats, self.normalized_adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), output

    def test(self, model, weights):
        model.load_state_dict(weights)
        return self.evaluate(model, self.test_mask)






