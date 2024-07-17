from copy import deepcopy
from opengsl.module.model.bmgcn import BMGCN
from opengsl.utils.utils import one_hot
from opengsl.module.encoder import MLPEncoder, GNNEncoder_OpenGSL
import torch
import time
from .solver import Solver
import torch.nn.functional as F


class BMGCNSolver(Solver):
    '''
        A solver to train, evaluate, test BMGCN in a run.

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
        >>> conf = opengsl.config.load_conf('bmgcn', 'cora')
        >>>
        >>> solver = BMGCNSolver(conf, dataset)
        >>> # Conduct a experiment run.
        >>> acc, new_structure = solver.run_exp(split=0, debug=True)
        '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "bmgcn"
        print("Solver Version : [{}]".format("bmgcn"))
        self.adj = self.adj.to_dense()
        self.adj = self.adj + torch.eye(self.adj.shape[0], device=self.device) * self.conf.self_loop


    def learn_nc(self, debug=False):
        '''
        Learning process of BMGCN.

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
        labels_oneHot = one_hot(self.labels)

        # pretrain
        print('Pretrain')
        for epoch in range(self.conf.training['n_epochs_mlp']):
            t0 = time.time()
            improve = ''
            self.model_mlp.train()
            self.optimizer_mlp.zero_grad()
            logits = self.model_mlp(self.feats)
            train_loss = self.loss_fn(logits[self.train_mask], self.labels[self.train_mask])
            train_acc = self.metric(self.labels[self.train_mask].cpu().numpy(), logits[self.train_mask].detach().cpu().numpy())
            train_loss.backward()
            self.optimizer_mlp.step()

            self.model_mlp.eval()
            logits = self.model_mlp(self.feats)
            val_loss = self.loss_fn(logits[self.val_mask], self.labels[self.val_mask])
            val_acc = self.metric(self.labels[self.val_mask].cpu().numpy(), logits[self.val_mask].detach().cpu().numpy())

            if val_acc >= self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = val_loss
                self.result['valid'] = val_acc
                self.result['train'] = train_acc
                self.mlp_weights = deepcopy(self.model_mlp.state_dict())

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, train_loss.item(), train_acc, val_loss, val_acc, improve))
        self.model_mlp.load_state_dict(self.mlp_weights)

        # train
        model_gcn = GNNEncoder_OpenGSL(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'], self.conf.model['dropout'], spmm_type=0)
        model_cpgnn = BMGCN(self.num_targets, self.model_mlp, model_gcn, [1, 1], self.conf.enhance, self.device)
        model_cpgnn.to(self.device)

        # all_params = model_cpgnn.parameters()
        # no_decay = []
        # for pname, p in model_cpgnn.named_parameters():
        #     if pname == 'H' or pname[-4:] == 'bias':
        #         # bias没有l2正则
        #         no_decay += [p]
        # params_id = list(map(id, no_decay))
        # other_params = list(filter(lambda p: id(p) not in params_id, all_params))
        # optimizer_cpgnn = torch.optim.Adam([
        #     {'params': no_decay},
        #     {'params': other_params, 'weight_decay': self.conf.training['weight_decay']}],
        #     lr=self.conf.training['lr'])

        optimizer_cpgnn = torch.optim.Adam(model_cpgnn.parameters(), weight_decay=self.conf.training['weight_decay'], lr=self.conf.training['lr'])
        for epoch in range(self.conf.training['n_epochs']):
            t0 = time.time()
            improve = ''
            model_cpgnn.train()
            optimizer_cpgnn.zero_grad()
            logits, train_loss, _, _, _ = model_cpgnn(self.feats, self.adj, self.train_mask, self.labels[self.train_mask], labels_oneHot,
                                                      self.train_mask)
            train_acc = self.metric(self.labels[self.train_mask].cpu().numpy(), logits[self.train_mask].detach().cpu().numpy())
            train_loss.backward()
            optimizer_cpgnn.step()

            model_cpgnn.eval()
            logits, val_loss, _, _, _ = model_cpgnn(self.feats, self.adj, self.val_mask, self.labels[self.val_mask], labels_oneHot, self.train_mask)
            val_acc = self.metric(self.labels[self.val_mask].cpu().numpy(), logits[self.val_mask].detach().cpu().numpy())
            flag, flag_earlystop = self.recoder.add(val_loss, val_acc)

            if flag:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = val_loss
                self.result['valid'] = val_acc
                self.result['train'] = train_acc
                self.bmgcn_weights = deepcopy(model_cpgnn.state_dict())
            elif flag_earlystop:
                break

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, train_loss.item(), train_acc, val_loss, val_acc, improve))

        model_cpgnn.load_state_dict(self.bmgcn_weights)
        model_cpgnn.eval()
        logits, _, H, Q, emb = model_cpgnn(self.feats, self.adj, self.test_mask, self.labels[self.test_mask], labels_oneHot, self.train_mask)
        test_acc = self.metric(self.labels[self.test_mask].cpu().numpy(), logits[self.test_mask].detach().cpu().numpy())
        self.result['test'] = test_acc
        return self.result, self.adjs

    def evaluate(self, test_mask):
        '''
        Evaluation procedure of GRCN.

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
            output, adjs = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), adjs

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model_mlp = MLPEncoder(self.feats.shape[1], self.conf.model['n_hidden'], self.num_targets, 2, self.conf.dropout_mlp, use_bn=False).to(self.device)
        self.optimizer_mlp = torch.optim.Adam(self.model_mlp.parameters(), lr=self.conf.training['lr'],
                                              weight_decay=self.conf.training['weight_decay'])
