import torch
import dgl
import copy
import math
import heapq
import numba as nb
import numpy as np


def add_knn(k, node_embed, edge_index):
    # 这里用cosine，和论文中不一致
    knn_g = dgl.knn_graph(node_embed,
                          k,
                          algorithm='bruteforce-sharemem',
                          dist='cosine')
    knn_g = dgl.add_reverse_edges(knn_g)
    knn_edge_index = knn_g.edges()
    knn_edge_index = torch.cat(
        (knn_edge_index[0].reshape(1, -1), knn_edge_index[1].reshape(1, -1)),
        dim=0)
    knn_edge_index = knn_edge_index.t()
    edge_index_2 = torch.cat((edge_index, knn_edge_index), dim=0)
    edge_index_2 = torch.unique(edge_index_2, dim=0)
    return edge_index_2


def calc_e1(adj: torch.Tensor):
    adj = adj - torch.diag_embed(torch.diag(adj))
    degree = adj.sum(dim=1)
    vol = adj.sum()
    idx = degree.nonzero().reshape(-1)
    g = degree[idx]
    return -((g / vol) * torch.log2(g / vol)).sum()


def get_adj_matrix(node_num, edge_index, weight) -> torch.Tensor:
    adj_matrix = torch.zeros((node_num, node_num))
    adj_matrix[edge_index.t()[0], edge_index.t()[1]] = weight.float()
    adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))  #去除对角线
    return adj_matrix


def get_weight(node_embedding, edge_index):
    node_num = node_embedding.shape[0]
    links = node_embedding[edge_index]
    weight = []
    for i in range(links.shape[0]):
        # print(links[i])
        # weight.append(torch.corrcoef(links[i])[0, 1])
        weight.append(np.corrcoef(links[i].cpu())[0, 1])
    weight = torch.tensor(weight) + 1
    weight[torch.isnan(weight)] = 0
    M = weight.mean() / (2 * node_num)
    weight = weight + M
    return weight


def knn_maxE1(edge_index: torch.Tensor, node_embedding: torch.Tensor):
    old_e1 = 0
    node_num = node_embedding.shape[0]
    k = 1
    while k < 50:
        edge_index_k = add_knn(k, node_embedding, edge_index)
        weight = get_weight(node_embedding, edge_index_k)
        # e1 = calc_e1(edge_index_k, weight)
        adj = get_adj_matrix(node_num, edge_index_k, weight)
        e1 = calc_e1(adj)
        if e1 - old_e1 > 0.1:
            k += 5
        elif e1 - old_e1 > 0.01:
            k += 3
        elif e1 - old_e1 > 0.001:
            k += 1
        else:
            break
        old_e1 = e1
    print(f'max1SE k: {k}')
    return k


def get_id():
    i = 0
    while True:
        yield i
        i += 1


def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    node_vol = []
    for i in range(g_num_nodes):
        n_v = 0
        adj = set()
        for j in range(g_num_nodes):
            if adj_matrix[i, j] != 0:
                n_v += adj_matrix[i, j]
                VOL += adj_matrix[i, j]
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return g_num_nodes, VOL, node_vol, adj_table


@nb.jit(nopython=True)
def cut_volume(adj_matrix, p1, p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i], p2[j]]
            if c != 0:
                c12 += c
    return c12


def LayerFirst(node_dict, start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h, node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID,
                                 partition=new_partition,
                                 children={id1, id2},
                                 g=g,
                                 vol=v,
                                 child_h=child_h,
                                 child_cut=cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(
        node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([
                node_dict[f_c].child_h for f_c in node_dict[parent_id].children
            ])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break


def child_tree_deepth(node_dict, nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth += 1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1, p_node):
    a = node1.child_cut
    v1 = node1.vol + 1
    v2 = p_node.vol + 1
    return a * math.log2(v2 / v1)


def CombineDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol + 1
    v2 = node2.vol + 1
    g1 = node1.g + 1
    g2 = node2.g + 1
    v12 = v1 + v2
    return ((v1 - g1) * math.log2(v12 / v1) + (v2 - g2) * math.log2(v12 / v2) -
            2 * cut_v * math.log2(g_vol / v12)) / g_vol


class PartitionTreeNode:
    def __init__(self,
                 ID,
                 partition,
                 vol,
                 g,
                 children: set = None,
                 parent=None,
                 child_h=0,
                 child_cut=0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h  #不包括该节点的子树高度
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__,
                                    self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}".format(k, getattr(self, k))
                        for k in self.__dict__.keys())


class PartitionTree:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(
            adj_matrix)
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()

    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = self.node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID,
                                          partition=[vertex],
                                          g=v,
                                          vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)

    def build_sub_leaves(self, node_list, p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        for vertex in node_list:
            ori_ent += -(self.tree_node[vertex].g / self.VOL)\
                       * math.log2((self.tree_node[vertex].vol+1)/(p_vol+1))
            sub_n = set()
            vol = 0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex, vertex_n]
                if c != 0:
                    vol += c
                    sub_n.add(vertex_n)
            sub_leaf = PartitionTreeNode(ID=vertex,
                                         partition=[vertex],
                                         g=vol,
                                         vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n

        return subgraph_node_dict, ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            ori_en += -(node.g / g_vol) * math.log2((node.vol + 1) / g_vol)
            new_n = set()
            for nei in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.add(nei)
            self.adj_table[node_id] = new_n

            new_node = PartitionTreeNode(ID=node_id,
                                         partition=node.partition,
                                         vol=node.vol,
                                         g=node.g,
                                         children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en

    def entropy(self, node_dict=None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id, node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol + 1
                node_g = node.g
                node_p_vol = node_p.vol + 1
                ent += -(node_g / self.VOL) * math.log2(node_vol / node_p_vol)
        return ent

    def __build_k_tree(
        self,
        g_vol,
        nodes_dict: dict,
        k=None,
    ):
        min_heap = []
        cmp_heap = []
        nodes_ids = nodes_dict.keys()
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v = self.adj_matrix[n1.partition[0],
                                                n2.partition[0]]
                    else:
                        cut_v = cut_volume(self.adj_matrix,
                                           p1=np.array(n1.partition),
                                           p2=np.array(n2.partition))
                    diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v,
                                        g_vol)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)
        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1].union(
                self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            #compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap, [
                    CompressDelta(nodes_dict[id1], nodes_dict[new_id]), id1,
                    new_id
                ])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap, [
                    CompressDelta(nodes_dict[id2], nodes_dict[new_id]), id2,
                    new_id
                ])
            unmerged_count -= 1

            for ID in self.adj_table[new_id]:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    cut_v = cut_volume(self.adj_matrix, np.array(n1.partition),
                                       np.array(n2.partition))

                    new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id],
                                            cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, ID, new_id, cut_v))
        root = new_id

        if unmerged_count > 1:
            #combine solitary node
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h
                               for i in unmerged_nodes]) + 1

            new_id = next(self.id_g)
            new_node = PartitionTreeNode(ID=new_id,
                                         partition=list(nodes_ids),
                                         children=unmerged_nodes,
                                         vol=g_vol,
                                         g=0,
                                         child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [
                        CompressDelta(nodes_dict[i], nodes_dict[new_id]), i,
                        new_id
                    ])
            root = new_id

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = CompressDelta(nodes_dict[e[1]],
                                                 nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = CompressDelta(nodes_dict[e[1]],
                                                 nodes_dict[p_id])
                heapq.heapify(cmp_heap)
        return root

    def check_balance(self, node_dict, root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict, c)

    def single_up(self, node_dict, node_id):
        new_id = next(self.id_g)
        p_id = node_dict[node_id].parent
        grow_node = PartitionTreeNode(ID=new_id,
                                      partition=node_dict[node_id].partition,
                                      parent=p_id,
                                      children={node_id},
                                      vol=node_dict[node_id].vol,
                                      g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        for i in self.adj_table[node_id]:
            self.adj_table[i].add(new_id)

    def root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0, None, None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(g_vol=g_vol,
                                       nodes_dict=subgraph_node_dict,
                                       k=2)
        self.check_balance(subgraph_node_dict, new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(
            self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self, sub_node_dict, sub_root_id, node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict, sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(
                    (node.vol + 1) / (node_p.vol + 1))
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(
                    (node.vol + 1) / (node_p.vol + 1))
        return ent

    def leaf_up(self):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) == 1:
                id_mapping[node_id] = None
            if len(sub_nodes) == 2:
                id_mapping[node_id] = None
            if len(sub_nodes) >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict, ori_ent = self.build_sub_leaves(
                    sub_nodes, candidate_node.vol)
                sub_root = self.__build_k_tree(g_vol=sub_g_vol,
                                               nodes_dict=subgraph_node_dict,
                                               k=2)
                self.check_balance(subgraph_node_dict, sub_root)
                new_ent = self.leaf_up_entropy(subgraph_node_dict, sub_root,
                                               node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta, id_mapping, h1_new_child_tree

    def leaf_up_update(self, id_mapping, leaf_up_dict):
        for node_id, h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node, i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1

    def root_down_update(self, new_id, root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode='v2'):
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k)
        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=2)
            self.check_balance(self.tree_node, self.root_id)

            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2

            flag = 0
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                    root_down_delta, new_id, root_down_dict = self.root_down_delta(
                    )

                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                elif flag == 2:
                    root_down_delta, new_id, root_down_dict = self.root_down_delta(
                    )
                else:
                    raise ValueError

                if leaf_up_delta < root_down_delta:
                    # print('root down')
                    # root down update and recompute root down delta
                    flag = 2
                    self.root_down_update(new_id, root_down_dict)

                else:
                    # leaf up update
                    # print('leave up')
                    flag = 1
                    # print(self.tree_node[self.root_id].child_h)
                    self.leaf_up_update(id_mapping, leaf_up_dict)
                    # print(self.tree_node[self.root_id].child_h)

                    # update root down leave nodes' children
                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items(
                        ):
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[
                                    root_down_id].children
        count = 0
        for _ in LayerFirst(self.tree_node, self.root_id):
            count += 1
        assert len(self.tree_node) == count

    def get_community(self):
        '''
        需要先进行编码树的构建 build_coding_tree
        k: k维结构熵
        返回tensor类型,只返回第二层的社区划分
        '''
        root_id = self.root_id
        partition_id = self.tree_node[root_id].children
        community = torch.zeros(len(self.adj_matrix[0]), dtype=torch.int32)
        community_id = 0
        for e in partition_id:
            partition_node = torch.tensor(self.tree_node[e].partition)
            community[partition_node] = community_id
            community_id += 1
        return community

    def get_community_3(self):
        '''
        需要先进行编码树的构建 build_coding_tree
        k: k维结构熵
        返回第三层的划分
        '''
        root_id = self.root_id
        partition_id = self.tree_node[root_id].children
        partition_2 = set()
        for e in partition_id:
            if self.tree_node[e].children == None:
                partition_2.add(e)
            else:
                partition_2.symmetric_difference_update(
                    self.tree_node[e].children)
        partition_id = partition_2
        community = torch.zeros(len(self.adj_matrix[0]), dtype=torch.int32)
        community_id = 0
        for e in partition_id:
            partition_node = torch.tensor(self.tree_node[e].partition)
            community[partition_node] = community_id
            community_id += 1
        return community

    def deduct_se(self, leaf_id, root_id=None):
        '''
        需先调用build_code_tree构建编码树!
        root_id = None时表示根节点为编码树根节点
        '''
        node_dict = self.tree_node
        path_id = [leaf_id]
        current_id = leaf_id
        while True:
            parent_id = node_dict[current_id].parent
            if parent_id == root_id:
                break
            if node_dict[parent_id].partition == node_dict[
                    current_id].partition:
                current_id = parent_id
                path_id[-1] = current_id
                continue
            path_id.append(parent_id)
            current_id = parent_id
        if root_id == None:
            path_id = path_id[0:-1]  #去除根节点
        g = []
        vol = []
        parent_vol = []
        for e in path_id:
            g.append(node_dict[e].g)
            vol.append(node_dict[e].vol)
            parent_vol.append(node_dict[node_dict[e].parent].vol)
        g = torch.tensor(g)
        vol = torch.tensor(vol) + 1
        parent_vol = torch.tensor(parent_vol) + 1
        deduct_se = -(g / self.VOL * torch.log2(vol / parent_vol)).sum()
        return deduct_se

    def LCA(self):
        node_dict = self.tree_node
        root_id = self.root_id
        tree_node_num = max(node_dict.keys()) + 1
        first = torch.zeros(tree_node_num)
        height = torch.zeros(tree_node_num)
        visited = torch.zeros(tree_node_num)
        euler = []
        stack = [root_id]
        h = 0
        while stack:
            node_id = stack.pop()
            euler.append(node_id)
            if visited[node_id]:
                h -= 1
                continue
            visited[node_id] = True
            height[node_id] = h
            h += 1
            first[node_id] = len(euler) - 1
            child = node_dict[node_id].children
            if child is None:
                continue
            child = list(child)
            if len(child) == 1 and node_dict[
                    child[0]].partition == node_dict[node_id].partition:
                child = child[0]
                while True:
                    first[child] = first[node_id]
                    child = node_dict[child].children
                    if child is None:
                        break
                    child = list(child)[0]
                continue
            for e in child[::-1]:
                stack.append(node_id)
                stack.append(e)
            # stack.append(child[0])
        self.first = first
        self.height = height
        self.euler = euler

    def query_LCA(self, id1, id2):
        tmp = torch.tensor([self.first[id1], self.first[id2]])
        begin, end = tmp.min(), tmp.max()
        interval = self.euler[begin.int():end.int() + 1]
        height_interval = self.height[interval]
        return interval[height_interval.argmin().int()]


def get_community(code_tree: PartitionTree):
    node_dict = code_tree.tree_node
    root_id = code_tree.root_id
    tree_node_num = max(node_dict.keys()) + 1
    isleaf = torch.zeros(tree_node_num, dtype=torch.bool)
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        child = node_dict[node_id].children
        if child is None:
            isleaf[node_id] = True
            continue
        child = list(child)
        for e in child[::-1]:
            stack.append(e)
        # stack.append(child[0])
    community = []
    for current_id in range(tree_node_num):
        if isleaf[current_id]:
            while True:
                parent_id = node_dict[current_id].parent
                if node_dict[parent_id].partition == node_dict[
                        current_id].partition:
                    isleaf[parent_id] = True
                    current_id = parent_id
                break
    for e in node_dict.keys():
        if not isleaf[e]:
            community.append(e)
    return community, isleaf


def get_sedict(community: list, code_tree: PartitionTree):
    node_dict = code_tree.tree_node
    se_dict = {}
    for community_id in community:
        node_list = list(node_dict[community_id].children)
        se = torch.zeros(len(node_list))
        for i, e in enumerate(node_list):
            e = node_dict[e]
            e: PartitionTreeNode
            se[i] = -(e.g / code_tree.VOL) * torch.log2(
                torch.tensor(
                    (e.vol + 1) /
                    (node_dict[e.parent].vol + 1))) + code_tree.deduct_se(
                        community_id, None)
            # se[i] = -(e.g / code_tree.VOL) * torch.log2(
            #     torch.tensor((e.vol + 1) / (node_dict[e.parent].vol + 1)))
        se = torch.softmax(se.float(), dim=0)
        se_dict[community_id] = se
    return se_dict


def select_link(community_id: int, code_tree: PartitionTree,
                isleaf: torch.Tensor, se_dict):
    node_dict = code_tree.tree_node
    node_list = list(node_dict[community_id].children)
    node_dict = code_tree.tree_node
    se = se_dict[community_id]
    id1, id2 = torch.multinomial(se, num_samples=2, replacement=True)
    link_id1 = node_list[id1]
    link_id2 = node_list[id2]
    link_id = [link_id1, link_id2]
    return link_id


def select_leaf(node_id, code_tree: PartitionTree, isleaf: torch.Tensor,
                se_dict):
    # print(node_id)
    node_dict = code_tree.tree_node
    while not isleaf[node_id]:
        node_list = list(node_dict[node_id].children)
        if len(node_list) > 1:  #避免只有一个字节点的非叶子节点
            se = se_dict[node_id]
            # print(se)
            id = torch.multinomial(se, num_samples=1, replacement=False)
            node_id = node_list[id]
        node_id = node_list[0]
    return (node_dict[node_id].partition)[0]


def reshape(community: list, code_tree: PartitionTree, isleaf: torch.Tensor,
            k):
    se_dict = {}
    edge_index = []
    node_dict = code_tree.tree_node
    # for k, v in code_tree.tree_node.items():
    #     print(k, v.__dict__)
    se_dict = get_sedict(community, code_tree)

    for community_id in community:
        node_list = list(node_dict[community_id].children)
        if len(node_list) == 1:
            continue
        prefer_edge_num = round(k * len(node_list))
        for i in range(prefer_edge_num):
            id1, id2 = select_link(community_id, code_tree, isleaf, se_dict)
            edge_index.append([
                select_leaf(id1, code_tree, isleaf, se_dict),
                select_leaf(id2, code_tree, isleaf, se_dict)
            ])
    edge_index = torch.tensor(edge_index)
    edge_index = torch.cat((edge_index, torch.flip(edge_index, dims=[1])),
                              dim=0)
    edge_index = torch.unique(edge_index, dim=0)
    return edge_index.t()
