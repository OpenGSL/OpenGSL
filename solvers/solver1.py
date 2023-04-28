import torch
from utils.utils import normalize_sp_tensor
from copy import deepcopy
import time
from utils.utils import accuracy
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


class Solver:
    def __init__(self, conf, dataset):
        self.conf = conf
        self.device = torch.device('cuda')
        self.n_nodes = dataset.n_nodes
        self.dim_feats = dataset.dim_feats
        self.num_targets = dataset.num_targets
        self.model = None
        self.loss_fn = F.binary_cross_entropy_with_logits if self.num_targets == 1 else F.cross_entropy
        self.metric = roc_auc_score if self.num_targets == 1 else accuracy
        self.model = None

        self.feats = dataset.feats
        self.adj = dataset.adj
        self.labels = dataset.labels
        self.train_masks = dataset.train_masks
        self.val_masks = dataset.val_masks
        self.test_masks = dataset.test_masks

    def prepare(self, split):
        if split is None:
            print('split set to default 0.')
            split=0
        assert split<len(self.train_masks), 'error, split id is larger than number of splits'
        self.train_mask = self.train_masks[split]
        self.val_mask = self.val_masks[split]
        self.test_mask = self.test_masks[split]
        self.reset()

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)


class TaskSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.normalize = normalize_sp_tensor if self.conf.dataset['normalize'] else lambda x, y: x

    def learn(self, split=None, debug=False):
        self.prepare(split)

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output = self.model(self.input_distributer())
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.val_mask)

            # save
            if loss_val < self.best_val_loss:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())

            # print
            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_distributer())
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def reset(self):
        self.model = self.get_model().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                         weight_decay=self.conf.training['weight_decay'])
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}
        self.start_time = time.time()
        self.normalized_adj = self.normalize(self.adj, add_loop=self.conf.dataset['add_loop'])


class GSLSolver(Solver):

    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def reset(self):
        self.get_model()
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.weights = None
        self.best_graph = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}
        self.start_time = time.time()






