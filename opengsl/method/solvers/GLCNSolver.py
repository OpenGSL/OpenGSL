from copy import deepcopy
import torch
import time
from .solver import Solver
from opengsl.method.models.glcn import GLCN


class GLCNSolver(Solver):
    '''
        A solver to train, evaluate, test GLCN in a run.

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
        >>> import opengsl.data.dataset
        >>> dataset = opengsl.data.Dataset('cora', feat_norm=True)
        >>> # load config file
        >>> import opengsl.config.load_conf
        >>> conf = opengsl.config.load_conf('grcn', 'cora')
        >>>
        >>> solver = GLCNSolver(conf, dataset)
        >>> # Conduct a experiment run.
        >>> acc, new_structure = solver.run_exp(split=0, debug=True)
        '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "grcn"
        print("Solver Version : [{}]".format("grcn"))
        edge_index = self.adj.coalesce().indices().cpu()
        loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
        edges = torch.cat([edge_index, loop_edge_index], dim=1)
        self.adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [self.n_nodes, self.n_nodes]).to(self.device).coalesce()


    def learn(self, debug=False):
        '''
        Learning process of GLCN.

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
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()

            # forward and backward
            output, _, others = self.model(self.feats, self.adj)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            loss_train += others['loss']
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            self.optim.zero_grad()
            loss_train.backward()
            self.optim.step()


            # Evaluate
            loss_val, acc_val, adjs = self.evaluate(self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            # save
            if flag:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
                self.adjs['final'] = adjs['final'].detach().clone()
            elif flag_earlystop:
                break

            # print

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.adjs

    def evaluate(self, test_mask):
        '''
        Evaluation procedure of GLCN.

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
            output, adjs, _ = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), adjs

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = GLCN(self.dim_feats, self.num_targets, self.conf).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                       weight_decay=self.conf.training['weight_decay'])