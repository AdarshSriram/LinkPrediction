import torch
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, negative_sampling
from torch_sparse import sum as sparse_sum
from adamic_utils import get_A, AA, get_pos_neg_edges
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.sparse
import time
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
from networkx.generators.community import stochastic_block_model
from networkx.generators.random_graphs import fast_gnp_random_graph
from networkx.classes.function import is_weighted, number_of_nodes
from networkx.convert import to_edgelist

hits = {
    "collab": [10, 50, 100],
    "ddi": [10, 20, 30],
    "block": [10, 20, 50]
}


def load_ogbl_collab():
    dataset = PygLinkPropPredDataset(name='ogbl-collab')
    data = dataset[0]
    edge_index = data.edge_index
    edge_weight = torch.ones(data.edge_index.size(1))

    if "edge_weight" in data:
        edge_weight = data.edge_weight.view(-1)

    split_edge = dataset.get_edge_split()
    # for key in split_edge:
    #     print(key)
    #     print(split_edge[key].keys())
    #     print()
    # print(split_edge['train']['edge'].size())
    # print(split_edge['train']['edge'])
    # print()
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    # print()
    # print()
    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    data.full_adj_t = data.adj_t
    # print(split_edge.keys())
    for key in split_edge:
        print(key)
        print(split_edge[key].keys())
        print()
    return edge_index, edge_weight, split_edge, data


def load_ogbl_ddi():
    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    print(dataset)
    data = dataset[0]
    print(data)

    # had to do this for ddi for some reason, for it to work
    data.num_nodes = data.num_nodes
    print(data.x)
    edge_index = data.edge_index
    edge_weight = torch.ones(data.edge_index.size(1))

    if "edge_weight" in data:
        edge_weight = data.edge_weight.view(-1)

    split_edge = dataset.get_edge_split()
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    data.full_adj_t = data.adj_t

    return edge_index, edge_weight, split_edge, data


def load_data_stochastic_block():
    p = 0.2
    q = 0.01
    gnp = stochastic_block_model([40, 40, 40, 40], [[p, q, q, q], [q, p, q, q], [q, q, p, q], [q, q, q, p]])
    # gnp = fast_gnp_random_graph(4000, 0.5)
    data = from_networkx(gnp)
    data.num_nodes = number_of_nodes(gnp)
    print("num nodes", data.num_nodes)
    edge_index = data.edge_index
    print(edge_index, np.shape(edge_index))
    edge_weight = torch.ones(edge_index.size(1))
    print("edge weight", edge_weight, np.shape(edge_weight))

    if is_weighted(gnp):
        edge_weight = data.edge_weight.view(-1)

    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    data.full_adj_t = data.adj_t

    split_edge = dict()
    split_edge['train'] = {}
    split_edge['valid'] = {}
    split_edge['test'] = {}
    split_edge['eval_train'] = {}

    idxs = torch.randperm(edge_index.size(1))
    valid_size = int(0.1 * edge_index.size(1))
    test_size = int(0.1 * edge_index.size(1))
    train_size = edge_index.size(1) - test_size - valid_size

    split_edge['train']['edge'] = edge_index.T[idxs[:train_size]]
    split_edge['valid']['edge'] = edge_index.T[idxs[train_size:train_size+valid_size]]
    print("line 115, load_data.py")
    print(split_edge['valid']['edge'])
    print(split_edge['valid']['edge'].size())
    split_edge['test']['edge'] = edge_index.T[idxs[train_size+valid_size:]]

    train_ind_subset = torch.randperm(split_edge['train']['edge'].size(0))[
        :split_edge['valid']['edge'].size(0)]
    split_edge['eval_train']['edge'] = split_edge['train']['edge'][train_ind_subset]

    neg_edge = negative_sampling(
        edge_index, num_nodes=data.num_nodes, num_neg_samples=valid_size + test_size, method='dense')
    print(neg_edge.size())
    print(valid_size)
    split_edge['valid']['edge_neg'], split_edge['test']['edge_neg'] = torch.split(
        neg_edge.T, [valid_size, test_size])
    # split_edge['test']['edge_neg'] = neg_edge[valid_size:].T
    print("neg edge dims")
    print(split_edge['valid']['edge_neg'].size())

    return edge_index, edge_weight, split_edge, data


def load_data(dataset):
    if dataset == "collab":
        return load_ogbl_collab()
    elif dataset == "ddi":
        return load_ogbl_ddi()
    elif dataset == "block":
        return load_data_stochastic_block()
    else:
        print("Invalid dataset name: {dataset}")
        return
