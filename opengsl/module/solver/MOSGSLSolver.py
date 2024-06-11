import time
from copy import deepcopy
import os
import sys
from torch_geometric.data import Dataset as PygDataset, Data, Batch
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from opengsl.module.encoder import MLPEncoder, GNNEncoder
from opengsl.module.solver import Solver
from opengsl.utils.recorder import Recorder
from opengsl.module.model.mosgsl import MOSGSL
# from .label import LabelInformedGraphLearner
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F


class MOSGSLSolver(Solver):
    def __init__(self, conf, dataset):
        super(MOSGSLSolver, self).__init__(conf, dataset)
        self.model = MOSGSL(self.conf, self.dim_feats, self.n_classes).to(self.device)
        self.model.parsing.init_parsing(self.dataset)
        self.pretrain_flag = 'n_pretrain' in self.conf.training and self.conf.training['n_pretrain'] > 0
        if self.pretrain_flag:
            assert self.conf.use_gsl

    def learn_gc(self, debug=False):
        if isinstance(self.dataset.data_raw, PygDataset):
            train_dataset = self.dataset.data_raw[self.train_mask]
            test_dataset = self.dataset.data_raw[self.test_mask]
            val_dataset = self.dataset.data_raw[self.val_mask]
        elif isinstance(self.dataset.data_raw, list):
            train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
            test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
            val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]
        else:
            raise NotImplementedError

        if self.pretrain_flag:
            self.pretrain(debug=debug)
            self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
            if self.conf.mode == 'testtime':
                self.model.backbone.requires_grad_(False)
        if self.conf.use_motif:
            self.init_motif(train_dataset)

        train_loader = DataLoader(train_dataset, self.conf.training['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, self.conf.training['test_batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, self.conf.training['test_batch_size'], shuffle=False)

        for epoch in range(self.conf.training['n_epochs']):

            if self.conf.use_motif and epoch >= self.conf.training['n_motif_update_min'] and epoch % self.conf.training['n_motif_update_each'] == 0:
                if debug:
                    print('Motif Projection')
                self.update_motif(train_dataset)

            improve = ''
            t0 = time.time()
            loss_train = 0

            # forward and backward
            preds = []
            ground_truth = []
            self.model.train()
            for batch in train_loader:
                self.optim.zero_grad()
                batch = batch.to(self.device)
                out, loss_con = self.model(batch)
                loss = self.loss_fn(out, batch.y.view(-1))
                if self.conf.use_motif:
                    loss += loss_con * self.conf.training['lambda']
                loss.backward()
                self.optim.step()
                loss_train += loss.item() * batch.num_graphs
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:,1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_train = loss_train / len(train_loader.dataset)
            acc_train = self.metric(ground_truth, preds)

            # Evaluate
            preds = []
            ground_truth = []
            self.model.eval()
            loss_val = 0
            for batch in val_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    out, _ = self.model(batch)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                loss_val += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_val = loss_val / len(val_loader.dataset)
            acc_val = self.metric(ground_truth, preds)
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

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train, acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        # test
        preds = []
        ground_truth = []
        self.model.load_state_dict(self.weights)
        self.model.eval()
        loss_test = 0
        for batch in test_loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                out, _ = self.model(batch)
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:,1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
            loss_test += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
        preds = torch.vstack(preds).squeeze().numpy()
        ground_truth = torch.vstack(ground_truth).squeeze().numpy()
        loss_test = loss_test / len(test_loader.dataset)
        acc_test = self.metric(ground_truth, preds)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))
        # for tune
        if self.current_split == 0 and os.path.basename(sys.argv[0])[:4] == 'tune':
            if acc_test < self.conf.tune['expect']:
                raise ValueError

        if 'mode' in self.conf and self.conf.mode == 'pretrain':
            self.result = {}
            self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
            self.model.backbone.reset_parameters()
            self.model.parsing.requires_grad_(False)
            self.model.selecting.requires_grad_(False)
            self.model.gsl.requires_grad_(False)
            self.model.motif.requires_grad_(False)
            optim = torch.optim.Adam(self.model.backbone.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])

            if isinstance(self.dataset.data_raw, PygDataset):
                train_dataset = self.dataset.data_raw[self.train_mask]
                test_dataset = self.dataset.data_raw[self.test_mask]
                val_dataset = self.dataset.data_raw[self.val_mask]
            elif isinstance(self.dataset.data_raw, list):
                train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
                test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
                val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]
            else:
                raise NotImplementedError

            train_loader = DataLoader(train_dataset, self.conf.training['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, self.conf.training['test_batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, self.conf.training['test_batch_size'], shuffle=False)

            for epoch in range(self.conf.training['n_epochs']):

                improve = ''
                t0 = time.time()
                loss_train = 0

                # forward and backward
                preds = []
                ground_truth = []
                self.model.train()
                for batch in train_loader:
                    optim.zero_grad()
                    batch = batch.to(self.device)
                    out, loss_con = self.model(batch)
                    loss = self.loss_fn(out, batch.y.view(-1))
                    loss.backward()
                    optim.step()
                    loss_train += loss.item() * batch.num_graphs
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                preds = torch.vstack(preds).squeeze().numpy()
                ground_truth = torch.vstack(ground_truth).squeeze().numpy()
                loss_train = loss_train / len(train_loader.dataset)
                acc_train = self.metric(ground_truth, preds)

                # Evaluate
                preds = []
                ground_truth = []
                self.model.eval()
                loss_val = 0
                for batch in val_loader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        out, _ = self.model(batch)
                        pred = F.softmax(out, dim=1)
                        if self.conf.training['metric'] != 'acc':
                            pred = pred[:, 1].unsqueeze(1)
                        preds.append(pred.detach().cpu())
                        ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                    loss_val += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
                preds = torch.vstack(preds).squeeze().numpy()
                ground_truth = torch.vstack(ground_truth).squeeze().numpy()
                loss_val = loss_val / len(val_loader.dataset)
                acc_val = self.metric(ground_truth, preds)
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

                if debug:
                    print(
                        "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                            epoch + 1, time.time() - t0, loss_train, acc_train, loss_val, acc_val, improve))
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            # test
            preds = []
            ground_truth = []
            self.model.load_state_dict(self.weights)
            self.model.eval()
            loss_test = 0
            for batch in test_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    out, _ = self.model(batch)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                loss_test += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_test = loss_test / len(test_loader.dataset)
            acc_test = self.metric(ground_truth, preds)
            self.result['test'] = acc_test
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))

        return self.result, None

    def set_method(self):
        self.model.reset_parameters()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])

    def init_motif(self, dataset):
        train_loader = DataLoader(dataset, self.conf.training['test_batch_size'], shuffle=True)
        zs = torch.tensor([]).to(self.device)
        ys = torch.tensor([], dtype=torch.long).to(self.device)
        self.model.eval()
        for data in train_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _, z_sub, belonging, importance = self.model.forward_wo_gsl(data, return_sub=True)
            # 先得到子图的标签
            y_sub = data.y[belonging]
            if self.pretrain_flag:
                # 如果进行了pretrain，可以进行筛选
                logits = F.softmax(out, dim=-1)
                con_max, cls_max = logits.max(dim=1)
                confidence = cls_max[belonging]
                score = (importance + confidence) / 2
                rank = score.sort(descending=True)[1]
                con_subgraph_idx = rank[:int(rank.shape[0] * self.conf.motif['k1'])]
                zs = torch.cat([zs, z_sub[con_subgraph_idx]], dim=0)
                ys = torch.cat([ys, y_sub[con_subgraph_idx]], dim=0)
            else:
                zs = torch.cat([zs, z_sub], dim=0)
                ys = torch.cat([ys, y_sub], dim=0)
        self.model.motif.init_motif(zs, ys)

    def update_motif(self, dataset):
        train_loader = DataLoader(dataset, self.conf.training['batch_size'], shuffle=True)
        zs = torch.tensor([]).to(self.device)
        ys = torch.tensor([], dtype=torch.long).to(self.device)
        self.model.eval()
        for data in train_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _, z_sub, belonging, importance = self.model(data, return_sub=True)
            # 先得到子图的标签
            y_sub = data.y[belonging]
            # 筛选confidence足够大的图的子图
            logits = F.softmax(out, dim=-1)
            con_max, cls_max = logits.max(dim=1)
            confidence = cls_max[belonging]
            score = (importance + confidence) / 2
            rank = score.sort(descending=True)[1]
            con_subgraph_idx = rank[:int(rank.shape[0] * self.conf.motif['k1'])]
            zs = torch.cat([zs, z_sub[con_subgraph_idx]], dim=0)
            ys = torch.cat([ys, y_sub[con_subgraph_idx]], dim=0)
        self.model.motif.update_motif(zs, ys)

    def pretrain(self, debug=False):
        if debug:
            print('Pretraining Start')
        if isinstance(self.dataset.data_raw, PygDataset):
            train_dataset = self.dataset.data_raw[self.train_mask]
            test_dataset = self.dataset.data_raw[self.test_mask]
            val_dataset = self.dataset.data_raw[self.val_mask]
        elif isinstance(self.dataset.data_raw, list):
            train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
            test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
            val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]

        train_loader = DataLoader(train_dataset, self.conf.training['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, self.conf.training['test_batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, self.conf.training['test_batch_size'], shuffle=False)

        pretrain_backbone = 'pretrain_backbone' in self.conf.training and self.conf.training['pretrain_backbone']
        for epoch in range(self.conf.training['n_pretrain']):
            improve = ''
            t0 = time.time()
            loss_train = 0

            # forward and backward
            preds = []
            ground_truth = []
            self.model.train()
            for data in train_loader:
                self.optim.zero_grad()
                data = data.to(self.device)
                out, _ = self.model.forward_wo_gsl(data, only_backbone=pretrain_backbone)
                loss = self.loss_fn(out, data.y.view(-1))
                loss.backward()
                self.optim.step()
                loss_train += loss.item() * data.num_graphs
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:, 1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_train = loss_train / len(train_loader.dataset)
            acc_train = self.metric(ground_truth, preds)

            # Evaluate
            preds = []
            ground_truth = []
            self.model.eval()
            loss_val = 0
            for data in val_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    out, _ = self.model.forward_wo_gsl(data, only_backbone=pretrain_backbone)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(data.y.detach().cpu().unsqueeze(1))
                loss_val += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_val = loss_val / len(val_loader.dataset)
            acc_val = self.metric(ground_truth, preds)
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

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(epoch + 1, time.time() - t0, loss_train, acc_train, loss_val, acc_val, improve))
        print('Pretrain End')
        # test
        preds = []
        ground_truth = []
        self.model.load_state_dict(self.weights)
        self.model.eval()
        loss_test = 0
        for data in test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _ = self.model.forward_wo_gsl(data, only_backbone=pretrain_backbone)
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:, 1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
            loss_test += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
        preds = torch.vstack(preds).squeeze().numpy()
        ground_truth = torch.vstack(ground_truth).squeeze().numpy()
        loss_test = loss_test / len(test_loader.dataset)
        acc_test = self.metric(ground_truth, preds)
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))
        if self.current_split == 0 and os.path.basename(sys.argv[0])[:4] == 'tune':
            if acc_test < self.conf.tune['expect']:
                print('Below Expectation')
                raise ValueError