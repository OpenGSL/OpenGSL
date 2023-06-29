import opengsl
import time
from copy import deepcopy
import torch


class GSL_Model(torch.nn.Module):
    def __init__(self, in_dim, output_dim):
        super(GSL_Model, self).__init__()
        self.layer = torch.nn.Linear(in_dim, output_dim)
        """
        init
        """

    def forward(self, input, adj):
        x = self.layer(input)
        return x


class GSL(opengsl.method.Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for gsl to train, evaluate, test in a run.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        self.method_name = "gsl"
        print("Solver Version : [{}]".format("gsl"))

    def learn(self, debug=False):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output = self.model(self.feats, self.adj)

            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                improve = '*'
                self.weights = deepcopy(self.model.state_dict())
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))

        return self.result, 0

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)

    def set_method(self):
        self.model = GSL_Model(self.dim_feats, self.num_targets).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                 weight_decay=self.conf.training['weight_decay'])


if __name__ == "__main__":
    
    conf = opengsl.config.load_conf(path="./configs/gsl_cora.yaml")
    dataset = opengsl.data.Dataset("cora", n_splits=1, feat_norm=conf.dataset['feat_norm'])
    solver = GSL(conf,dataset)


    exp = opengsl.ExpManager(solver)
    exp.run(n_runs = 10)