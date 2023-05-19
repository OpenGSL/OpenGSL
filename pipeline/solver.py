import torch
from copy import deepcopy
import time
from utils.utils import accuracy
from utils.recorder import Recorder
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import wandb


class Solver:
    def __init__(self, conf, dataset):
        self.conf = conf
        self.device = torch.device('cuda')
        self.n_nodes = dataset.n_nodes
        self.dim_feats = dataset.dim_feats
        self.num_targets = dataset.num_targets
        self.n_classes = dataset.n_classes
        self.model = None
        self.loss_fn = F.binary_cross_entropy_with_logits if self.num_targets == 1 else F.cross_entropy
        self.metric = roc_auc_score if self.num_targets == 1 else accuracy
        self.model = None

        self.feats = dataset.feats
        self.adj = dataset.adj if self.conf.dataset['sparse'] else dataset.adj.to_dense()
        self.labels = dataset.labels
        self.train_masks = dataset.train_masks
        self.val_masks = dataset.val_masks
        self.test_masks = dataset.test_masks

    def run_exp(self, split=None, debug=False):
        self.set(split)
        return self.learn(debug)

    def set(self, split):
        '''
        This sets necessary members for a run.
        Parameters
        ----------
        split

        Returns
        -------

        '''
        if split is None:
            print('split set to default 0.')
            split=0
        assert split<len(self.train_masks), 'error, split id is larger than number of splits'
        self.train_mask = self.train_masks[split]
        self.val_mask = self.val_masks[split]
        self.test_mask = self.test_masks[split]
        self.total_time = 0
        self.best_val_loss = 10
        self.weights = None
        self.best_graph = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}
        self.start_time = time.time()
        self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
        self.set_method()

    def set_method(self):
        '''
        This sets model and other members, which is overrided for each method
        Returns
        -------

        '''
        self.model = None
        self.optim = None

    def learn(self, debug=False):
        '''
        This is the learning process of common gnns, which is overrided for special learning process

        Parameters
        ----------
        debug

        Returns
        -------

        '''
        if 'analysis' in self.conf and self.conf.analysis['flag']:
            if not ('sweep' in self.conf.analysis and self.conf.analysis['sweep']):
                wandb.init(config=self.conf,
                           project=self.conf.analysis['project'])
            wandb.define_metric("acc_val", summary="max")
            wandb.define_metric("loss_val", summary="min")
            wandb.define_metric("loss_train", summary="min")
            wandb.define_metric("acc_train", summary="max")

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output = self.model(self.input_distributer())
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.val_mask)
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

            # print
            if 'analysis' in self.conf and self.conf.analysis['flag']:
                wandb.log({'epoch':epoch+1,
                           'acc_val':acc_val,
                           'loss_val':loss_val,
                           'acc_train': acc_train,
                           'loss_train': loss_train})

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        if 'analysis' in self.conf and self.conf.analysis['flag']:
            wandb.log({'loss_test':loss_test, 'acc_test':acc_test})
            if not ('sweep' in self.conf.analysis and self.conf.analysis['sweep']):
                wandb.finish()
        return self.result, 0

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_distributer())
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def input_distributer(self):
        '''
        This distributes different input in "learn" for different methods, overrided for each method
        Returns
        -------

        '''
        return None

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)





