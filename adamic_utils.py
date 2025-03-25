import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.sparse as ssp
from torch_geometric.utils import to_undirected, negative_sampling

# Convert adjacency-list representation of a network to sparse (CSR) matrix
def get_A(adj, num_nodes):
    row, col,val = adj.coo()
    row = row.cpu().numpy().tolist()
    col = col.cpu().numpy().tolist()
    val = val.cpu().numpy().tolist()

    # print(row.size())
    # print(type(row))
    # print(len(row))
    # print(type(col))
    # print(val)
    
    A = ssp.csr_matrix((val, (row, col)), shape=(num_nodes, num_nodes))
    return A
    
# Compute Adamic-Adar heuristic score and corresponding edge indices. Input network should be inCSR format
def AA(A, edge_index, batch_size=2000):
    multiplier = 1 / np.log(A.sum(0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in (link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src.cpu()].multiply(A_[dst.cpu()]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

# Compute the common neighbours heuristic score given a networks adjacency list
def common_neighbours(adj, edge_index):
    cn = adj[edge_index[0]].to_torch_sparse_coo_tensor().mul(adj[edge_index[1]].to_torch_sparse_coo_tensor())
    if cn._nnz() == 0:
        return torch.zeros((cn.shape[0])).to("cuda:0")
    return torch.sparse.sum(cn, 1).to_dense()
    
# Helper to generate positive and negative edges
def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            # new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsampe
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    return pos_edge, neg_edge
