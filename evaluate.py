import torch
import os
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_sparse import sum as sparse_sum
from adamic_utils import get_A, AA, get_pos_neg_edges
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.sparse
import time
from scipy.sparse.linalg import inv
from scipy.sparse import eye
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from logger import Logger
#import pagerank as pg
from load_data import hits
import datetime

@torch.no_grad()
def test_gcn(model, data, split_edge, evaluator, batch_size, device, dataset):
    model.eval()

    pos_train_edge = split_edge['eval_train']['edge'].to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)

    pos_train_preds = []
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size)):
        edge = pos_train_edge[perm].t()
        pos_train_preds.append(model( data.x,edge, data.adj_t).cpu().squeeze())
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    
    pos_valid_preds = []
    for perm in tqdm(DataLoader(range(pos_valid_edge.size(0)), batch_size)):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds.append(model(  data.x,edge, data.adj_t).cpu().squeeze())
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in tqdm(DataLoader(range(neg_valid_edge.size(0)), batch_size)):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds.append(model( data.x,edge, data.adj_t).cpu().squeeze())
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in tqdm(DataLoader(range(pos_test_edge.size(0)), batch_size)):
        edge = pos_test_edge[perm].t()
        pos_test_preds.append(model( data.x,edge, data.full_adj_t).cpu().squeeze())
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in tqdm(DataLoader(range(neg_test_edge.size(0)), batch_size)):
        edge = neg_test_edge[perm].t()
        neg_test_preds.append(model( data.x,edge, data.full_adj_t).cpu().squeeze())
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    results = {}
    for K in hits[dataset]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def test_adamic(data, split_edge, evaluator, device, dataset):
    A_eval = get_A(data.adj_t, data.num_nodes)
    A = get_A(data.full_adj_t, data.num_nodes)
    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                 data.edge_index, 
                                                 data.num_nodes)
    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_train_pred = torch.ones(pos_train_edge.size(0))
    pos_valid_pred, pos_valid_edge = eval('AA')(A_eval, pos_val_edge)
    # print(pos_valid_edge.size())
    # print(pos_val_edge)
    neg_valid_pred, neg_valid_edge = eval('AA')(A_eval, neg_val_edge)
    # print(neg_val_edge.size())
    # print(neg_val_edge)
    pos_test_pred, pos_test_edge = eval('AA')(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval('AA')(A, neg_test_edge)

    results = {}
    for K in hits[dataset]:
        print(K)
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    return results

def test_katz(data, split_edge, evaluator, batch_size, device, dataset, collab=False):
    A = get_A(data.full_adj_t, data.num_nodes)
    A_train = get_A(data.adj_t, data.num_nodes)
    pos_train_edge = split_edge['eval_train']['edge']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    beta = 0.05
    if collab:
        H_train = beta * A_train
        for _ in range(2):
            H_train += beta * (A_train @ H_train)

        H = beta * A
        for _ in range(2):
            H += beta * (A @ H)
    else:
        H_train = inv(eye(data.num_nodes) - beta * A_train) - eye(data.num_nodes) 
        H = inv(eye(data.num_nodes) - beta * A) - eye(data.num_nodes) 
    
    pos_train_preds = []
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), 100)):
        edge = pos_train_edge[perm].t()
        pos_train_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_train_pred = torch.cat(pos_train_preds, dim=0)    
    
    pos_valid_preds = []
    for perm in tqdm(DataLoader(range(pos_valid_edge.size(0)), batch_size)):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in tqdm(DataLoader(range(neg_valid_edge.size(0)), batch_size)):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in tqdm(DataLoader(range(pos_test_edge.size(0)), batch_size)):
        edge = pos_test_edge[perm].t()
        pos_test_preds.append(torch.tensor(H[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in tqdm(DataLoader(range(neg_test_edge.size(0)), batch_size)):
        edge = neg_test_edge[perm].t()
        neg_test_preds.append(torch.tensor(H[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    results = {}
    for K in hits[dataset]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results