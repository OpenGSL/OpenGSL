import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
import numpy as np
from opengsl.module.encoder import GNNEncoder_OpenGSL, GNNEncoder


class Classification(nn.Module):
    def __init__(self, num_feature, cls_hid_1, num_class, dropout, pyg):
        super(Classification, self).__init__()
        self.num_class = num_class
        self.pyg = pyg
        if pyg == False:
            self.encoder_v1 = GNNEncoder_OpenGSL(n_feat=num_feature, n_hidden=cls_hid_1, n_class=num_class, dropout=dropout, weight_initializer='glorot', bias_initializer='zeros')
            self.encoder_v2 = GNNEncoder_OpenGSL(n_feat=num_feature, n_hidden=cls_hid_1, n_class=num_class, dropout=dropout, weight_initializer='glorot', bias_initializer='zeros')
            self.encoder_v = GNNEncoder_OpenGSL(n_feat=num_feature, n_hidden=cls_hid_1, n_class=num_class, dropout=dropout, weight_initializer='glorot', bias_initializer='zeros')
        else:
            self.encoder_v1 = GNNEncoder(n_feat=num_feature, n_hidden=cls_hid_1, n_class=num_class, dropout=dropout)
            self.encoder_v2 = GNNEncoder(n_feat=num_feature, n_hidden=cls_hid_1, n_class=num_class, dropout=dropout)
            self.encoder_v = GNNEncoder(n_feat=num_feature, n_hidden=cls_hid_1, n_class=num_class, dropout=dropout)

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, feat, view, flag):
        if not self.pyg:
            view = view.to_dense()
        if (self.num_class == 1):
            if flag == "v1":
                prob = F.sigmoid(self.encoder_v1(feat, view))
            elif flag == "v2":
                prob = F.sigmoid(self.encoder_v2(feat, view))
            elif flag == "v":
                prob = F.sigmoid(self.encoder_v(feat, view))
        else:
            if flag == "v1":
                prob = F.softmax(self.encoder_v1(feat, view), dim=1)
                # prob = F.sigmoid(self.encoder_v1(feat, view))
            elif flag == "v2":
                prob = F.softmax(self.encoder_v2(feat, view), dim=1)
                # prob = F.sigmoid(self.encoder_v2(feat, view))
            elif flag == "v":
                prob = F.softmax(self.encoder_v(feat, view), dim=1)
                # prob = F.sigmoid(self.encoder_v(feat, view))
        return prob


class Contrast:
    def __init__(self, tau):
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def cal(self, z1_proj, z2_proj):
        matrix_z1z2 = self.sim(z1_proj, z2_proj)
        matrix_z2z1 = matrix_z1z2.t()

        matrix_z1z2 = matrix_z1z2 / (torch.sum(matrix_z1z2, dim=1).view(-1, 1) + 1e-8)
        lori_v1v2 = -torch.log(matrix_z1z2.diag()+1e-8).mean()

        matrix_z2z1 = matrix_z2z1 / (torch.sum(matrix_z2z1, dim=1).view(-1, 1) + 1e-8)
        lori_v2v1 = -torch.log(matrix_z2z1.diag()+1e-8).mean()
        return (lori_v1v2 + lori_v2v1) / 2


class Fusion(nn.Module):
    def __init__(self, lam, alpha, name):
        super(Fusion, self).__init__()
        self.lam = lam
        self.alpha = alpha
        self.name = name

    def get_weight(self, prob):
        if len(prob.shape)==1:
            prob = torch.stack([prob, 1 - prob],dim=1)
        out, _ = prob.topk(2, dim=1, largest=True, sorted=True)
        fir = out[:, 0]
        sec = out[:, 1]
        w = torch.exp(self.alpha*(self.lam*torch.log(fir+1e-8) + (1-self.lam)*torch.log(fir-sec+1e-8)))
        return w

    def forward(self, v1, prob_v1, v2, prob_v2):
        w_v1 = self.get_weight(prob_v1)
        w_v2 = self.get_weight(prob_v2)
        beta_v1 = w_v1 / (w_v1 + w_v2)
        beta_v2 = w_v2 / (w_v1 + w_v2)
        if self.name not in ["citeseer", "digits", "polblogs", "cora"]:
            beta_v1 = beta_v1.diag().to_sparse()
            beta_v2 = beta_v2.diag().to_sparse()
            v = torch.sparse.mm(beta_v1, v1) + torch.sparse.mm(beta_v2, v2)
            return v
        else :
            beta_v1 = beta_v1.unsqueeze(1)
            beta_v2 = beta_v2.unsqueeze(1)
            v = beta_v1 * v1.to_dense() + beta_v2 * v2.to_dense()
            return v.to_sparse()


class GenView(nn.Module):
    def __init__(self, num_feature, hid, com_lambda, dropout, pyg):
        super(GenView, self).__init__()
        self.pyg = pyg
        if pyg == False:
            self.gen_gcn = GNNEncoder_OpenGSL(n_feat=num_feature, n_hidden=hid, n_class=hid, n_layers=1, dropout=0, weight_initializer='glorot', bias_initializer='zeros')
        else:
            self.gen_gcn = GNNEncoder(n_feat=num_feature, n_hidden=hid, n_class=hid, n_layers=1, dropout=0)
        #self.gen_gcn = GraphConvolution(num_feature, hid, dropout=0)
        self.gen_mlp = nn.Linear(2 * hid, 1)
        nn.init.xavier_normal_(self.gen_mlp.weight, gain=1.414)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.com_lambda = com_lambda
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, v_ori, feat, v_indices, num_node):
        emb = self.gen_gcn(feat, v_ori if self.pyg else v_ori.to_dense())
        f1 = emb[v_indices[0]]
        f2 = emb[v_indices[1]]
        ff = torch.cat([f1, f2], dim=-1)
        temp = self.gen_mlp(self.dropout(ff)).reshape(-1)
        
        z_matrix = torch.sparse.FloatTensor(v_indices, temp, (num_node, num_node))
        pi = torch.sparse.softmax(z_matrix, dim=1)
        gen_v = v_ori + self.com_lambda * pi 
        return gen_v


class View_Estimator(nn.Module):
    def __init__(self, num_feature, gen_hid, com_lambda_v1, com_lambda_v2, dropout, pyg, big):
        super(View_Estimator, self).__init__()
        self.v1_gen = GenView(num_feature, gen_hid, com_lambda_v1, dropout, pyg)
        self.v2_gen = GenView(num_feature, gen_hid, com_lambda_v2, dropout, pyg)
        if big:
            self.normalize = self.normalize1
        else:
            self.normalize = self.normalize2

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def normalize1(self, adj):
        return (adj + adj.t())

    def normalize2(self, mx):
        mx = mx + mx.t() + torch.eye(mx.shape[0]).to(mx.device).to_sparse()
        mx = mx.to_dense()
        rowsum = mx.sum(1) + 1e-6  # avoid NaN
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx.to_sparse()

    def forward(self, view1, view1_indices, view2, view2_indices, num_nodes, feats):
        new_v1 = self.normalize(self.v1_gen(view1, feats, view1_indices, num_nodes))
        new_v2 = self.normalize(self.v2_gen(view2, feats, view2_indices, num_nodes))
        return new_v1, new_v2


class MI_NCE(nn.Module):
    def __init__(self, num_feature, mi_hid_1, tau, pyg, big, batch):
        super(MI_NCE, self).__init__()
        self.pyg = pyg
        if pyg == False:
            self.gcn = GNNEncoder_OpenGSL(n_feat=num_feature, n_hidden=mi_hid_1, n_class=mi_hid_1, n_layers=1, act='nn.PReLU()', dropout=0, weight_initializer='glorot', bias_initializer='zeros')
            self.gcn1 = GNNEncoder_OpenGSL(n_feat=num_feature, n_hidden=mi_hid_1, n_class=mi_hid_1, act='nn.PReLU()', dropout=0, weight_initializer='glorot', bias_initializer='zeros')
            self.gcn2 = GNNEncoder_OpenGSL(n_feat=num_feature, n_hidden=mi_hid_1, n_class=mi_hid_1, act='nn.PReLU()', dropout=0, weight_initializer='glorot', bias_initializer='zeros')
        else:
            #print("pyg")
            self.gcn = GNNEncoder(n_feat=num_feature, n_hidden=mi_hid_1, n_class=mi_hid_1, n_layers=1, dropout=0, act='nn.PReLU()')
            self.gcn1 = GNNEncoder(n_feat=num_feature, n_hidden=mi_hid_1, n_class=mi_hid_1, n_layers=1, dropout=0, act='nn.PReLU()')
            self.gcn2 = GNNEncoder(n_feat=num_feature, n_hidden=mi_hid_1, n_class=mi_hid_1, n_layers=1, dropout=0, act='nn.PReLU()')
        self.proj = nn.Sequential(
            nn.Linear(mi_hid_1, mi_hid_1),
            nn.ELU(),
            nn.Linear(mi_hid_1, mi_hid_1)
        )
        self.con = Contrast(tau)
        self.big = big
        self.batch = batch

    def reset_parameters(self):
        self.gcn.reset_parameters()
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        for child in self.proj:
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, views, feat):
        v_emb = self.proj(self.gcn(feat, views[0] if self.pyg else views[0].to_dense()))
        v1_emb = self.proj(self.gcn1(feat, views[1] if self.pyg else views[1].to_dense()))
        v2_emb = self.proj(self.gcn2(feat, views[2] if self.pyg else views[2].to_dense()))
        # if dataset is so big, we will randomly sample part of nodes to perform MI estimation
        if self.big == True:
            idx = np.random.choice(feat.shape[0], self.batch, replace=False)
            idx.sort()
            v_emb = v_emb[idx]
            v1_emb = v1_emb[idx]
            v2_emb = v2_emb[idx]
            
        vv1 = self.con.cal(v_emb, v1_emb)
        vv2 = self.con.cal(v_emb, v2_emb)
        v1v2 = self.con.cal(v1_emb, v2_emb)

        return vv1, vv2, v1v2


class CoGSL(nn.Module):
    def __init__(self, num_feature, cls_hid_1, num_class, gen_hid, mi_hid_1, com_lambda_v1, com_lambda_v2, lam, alpha,
                 cls_dropout, ve_dropout, tau, pyg, big, batch, name):
        super(CoGSL, self).__init__()
        self.cls = Classification(num_feature, cls_hid_1, num_class, cls_dropout, pyg)
        self.ve = View_Estimator(num_feature, gen_hid, com_lambda_v1, com_lambda_v2, ve_dropout, pyg, big)
        self.mi = MI_NCE(num_feature, mi_hid_1, tau, pyg, big, batch)
        self.fusion = Fusion(lam, alpha, name)

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def get_view(self, view1, view1_indices, view2, view2_indices, num_nodes, feats):
        new_v1, new_v2 = self.ve(view1, view1_indices, view2, view2_indices, num_nodes, feats)
        return new_v1, new_v2

    def get_mi_loss(self, feat, views):
        mi_loss = self.mi(views, feat)
        return mi_loss

    def get_cls_loss(self, v1, v2, feat):
        prob_v1 = self.cls(feat, v1, "v1")
        prob_v2 = self.cls(feat, v2, "v2")
        logits_v1 = torch.log(prob_v1 + 1e-8)
        logits_v2 = torch.log(prob_v2 + 1e-8)
        return logits_v1.squeeze(1), logits_v2.squeeze(1), prob_v1.squeeze(1), prob_v2.squeeze(1)

    def get_v_cls_loss(self, v, feat):
        logits = torch.log(self.cls(feat, v, "v") + 1e-8)
        return logits.squeeze(1)

    def get_fusion(self, v1, prob_v1, v2, prob_v2):
        v = self.fusion(v1, prob_v1, v2, prob_v2)
        return v