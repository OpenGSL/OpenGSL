import torch
from models.jknet import JKNet
from utils.utils import normalize_sp_tensor, accuracy, set_seed, sample_mask
from copy import deepcopy
import time
from .solver import BaseSolver
import wandb


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("gcndense"))
        self.normalize = normalize_sp_tensor if self.conf.dataset['normalize'] else lambda x, y :x

    def train(self):
        if 'analysis' in self.conf and self.conf.analysis['flag']:
            wandb.init(config=self.cfg,
                       project=self.cfg.analysis['project'])
            wandb.define_metric("loss_val", summary="min")
            wandb.define_metric("acc_val", summary="max")
            wandb.define_metric("loss_train", summary="min")
            wandb.define_metric("acc_train", summary="max")
        model = JKNet(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'], self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'], self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'], self.conf.model['general'],self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        total_time = 0
        best_val_loss = 10
        weights = None
        result = {'train': 0, 'valid': 0, 'test': 0}
        normalized_adj = self.normalize(self.adj, self.conf.dataset['add_loop'])
        start_time = time.time()
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            model.train()
            optim.zero_grad()

            # forward and backward
            x, output = model((self.feats, normalized_adj,False))

            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            optim.step()

            # Evaluate
            loss_val, acc_val, _ = self.evaluate(model, self.val_mask, normalized_adj)
            loss_test, acc_test, _ = self.evaluate(model, self.test_mask, normalized_adj)

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
        loss_test, acc_test, _ = self.test(model, weights)
        result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        wandb.finish()
        return result

    def evaluate(self, model, test_mask, normalized_adj):
        model.eval()
        with torch.no_grad():
            x, output = model((self.feats, normalized_adj,False))
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), output

    def test(self, model, weights):
        model.load_state_dict(weights)
        normalized_adj = self.normalize(self.adj, self.conf.dataset['add_loop'])
        return self.evaluate(model, self.test_mask, normalized_adj)
