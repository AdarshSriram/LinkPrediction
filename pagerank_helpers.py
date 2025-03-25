import torch
from adamic_utils import get_A
import numpy as np
import scipy.sparse
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import random
import math
from tqdm import tqdm
import os

SAVED_DATA_DIR = "saved_data"
if not os.path.isdir(SAVED_DATA_DIR):
    os.mkdir(SAVED_DATA_DIR)

# Compute pageranks for the network using networkx lib, and write them to txt files
def get_pagerank(dataset, data):
    try:
        eval_g = scipy.sparse.load_npz(f"{SAVED_DATA_DIR}/{dataset}/evalG.npz")
        full_g = scipy.sparse.load_npz(f"{SAVED_DATA_DIR}/{dataset}/fullG.npz")
    except:    
        full_g = get_A(data.full_adj_t, data.num_nodes)
        eval_g = get_A(data.adj_t, data.num_nodes)
    if not os.path.isdir(SAVED_DATA_DIR+dataset):
        os.mkdir(f"{SAVED_DATA_DIR}/{dataset}")
    scipy.sparse.save_npz(f'{SAVED_DATA_DIR}/{dataset}/evalG.npz', eval_g)
    scipy.sparse.save_npz(f'{SAVED_DATA_DIR}/{dataset}/fullG.npz', full_g)

    full_net = nx.from_scipy_sparse_matrix(full_g)
    eval_net = nx.from_scipy_sparse_matrix(eval_g)
    full_pgrnk = pagerank(full_net)
    eval_pgrnk = pagerank(eval_net)
    sorted_pgrnks_full = sorted(full_pgrnk, key=full_pgrnk.get, reverse=True)
    sorted_pgrnks_eval = sorted(eval_pgrnk, key=eval_pgrnk.get, reverse=True)

    full_idx_f = open(f"{SAVED_DATA_DIR}/{dataset}/full_pgrnks.txt", "w")
    for x in sorted_pgrnks_full:
        full_idx_f.write(str(x) + "\n")
    full_idx_f.close()

    eval_idx_f = open(f"{SAVED_DATA_DIR}/{dataset}/eval_pgrnks.txt", "w")
    for x in sorted_pgrnks_eval:
        eval_idx_f.write(str(x) + "\n")
    eval_idx_f.close()

    return sorted_pgrnks_full, sorted_pgrnks_eval

def top_k_nodes(K, data, dataset):
    try:
        with open(f"{SAVED_DATA_DIR}/{dataset}/full_pgrnks.txt") as f1:
            full_pgrnk_indices = f1.read().splitlines()
        with open(f"{SAVED_DATA_DIR}/{dataset}/eval_pgrnks.txt") as f2:
            eval_pgrnk_indices = f2.read().splitlines()
    except:
        full_pgrnk_indices, eval_pgrnk_indices = get_pagerank(dataset, data)

    return full_pgrnk_indices[:K], eval_pgrnk_indices[:K]

def add_random_beacons(num_randoms, data, top_idxs_te, top_idxs_tr):
    n = data.num_nodes
    top_tr_set, top_te_set = set(top_idxs_tr), set(top_idxs_te)
    # test_randoms = tr_randoms = num_randoms
    while num_randoms > 0:
        rand_idx = random.randint(0,n)
        if rand_idx not in top_tr_set:
            top_idxs_tr.append(rand_idx)
            num_randoms -= 1

def katz_score(beta, data):
    A = get_A(data.full_adj_t, data.num_nodes)
    A_train = get_A(data.adj_t, data.num_nodes)

    # TODO: could change to full computation for ddi
    H_train = beta * A_train
    for _ in range(2):
        H_train += beta * (A_train @ H_train)

    H = beta * A
    for _ in range(2):
        H += beta * (A @ H)

    return H, H_train

# Augment nodes with heuristic scores
def augment(idx, Score, top_k_idxs):
    dist_vec = []
    for ind in top_k_idxs:
        dist_vec.append(Score[idx, int(ind)])
    return np.array(dist_vec)


def pagerank_augment(data, dataset, k_pg, katz_beta, random_split):
    num_randoms = math.floor(k_pg*random_split)
    num_imp = k_pg - num_randoms
    print(f"Beacon split: {num_imp} important nodes, {num_randoms} random nodes")
    print("Getting beacon nodes...")
    test_beacons, train_beacons = top_k_nodes(num_imp, data, dataset)
    if random_split > 0:
        add_random_beacons(num_randoms, data, test_beacons, train_beacons)

    print("Getting pairwise katz...")
    test_katz, train_katz = katz_score(katz_beta, data)

    print("Augmenting node features...")
    aug_tr = np.zeros((data.num_nodes, k_pg))
    aug_te = np.zeros((data.num_nodes, k_pg))
    for i in tqdm(range(data.num_nodes)):
        aug_tr[i] = augment(i, train_katz, train_beacons)
        aug_te[i] = augment(i, test_katz, train_beacons)
        # aug_te[i] = augment(i, test_katz, test_beacons)
    if data.x is None:
        augmented_tr = torch.Tensor(aug_tr)
        augmented_te = torch.Tensor(aug_te)
    else:
        augmented_tr = torch.hstack([data.x, torch.Tensor(aug_tr).to(torch.device("cpu"))])
        augmented_te = torch.hstack([data.x, torch.Tensor(aug_te).to(torch.device("cpu"))])

    return augmented_tr, augmented_te 
