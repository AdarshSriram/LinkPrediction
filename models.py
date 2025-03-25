import argparse
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling, to_undirected
from torch_sparse import sum as sparse_sum
from adamic_utils import get_A, AA, common_neighbours
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.sparse
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from logger import Logger
from load_data import load_data, hits
from evaluate import test_gcn, test_adamic, test_katz
from pagerank_helpers import pagerank_augment


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, aug):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        if aug:
            self.lins.append(torch.nn.Linear(in_channels+1, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, score=None):
        x = x_i * x_j
        if score != None:
            score = torch.reshape(score, (score.shape[0], 1)).to(
                torch.device("cpu"))
            x = torch.hstack((x, score))
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LinkGNN(torch.nn.Module):
    def __init__(self, emb, gnn, linkpred, aug=None):
        super(LinkGNN, self).__init__()
        self.gnn = gnn
        self.linkpred = linkpred
        self.emb = emb.float()
        self.aug = aug

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.linkpred.reset_parameters()
        if self.emb is not None:
            self.emb.reset_parameters()

    def forward(self, x, edges, adj):
        if x is None:
            x = self.emb.weight
        elif self.emb is not None:
            x = x.float()
            x = torch.cat([self.emb.weight, x], dim=1)
        x = x.float()
        h = self.gnn(x, adj)
        if self.aug == "cn":
            common_nbr_score = common_neighbours(adj, edges)
            return self.linkpred(h[edges[0]], h[edges[1]], common_nbr_score)
        elif self.aug == "aa":
            A_sparse = get_A(adj, adj.size(0))
            AA_score, _ = AA(A_sparse, edges)
            return self.linkpred(h[edges[0]], h[edges[1]], AA_score)
        elif self.aug == None:
            return self.linkpred(h[edges[0]], h[edges[1]])


def build_gcn(args, data, device, use_features, use_emb, aug=None):
    input_dim = 0
    emb = None
    if use_emb:
        emb = torch.nn.Embedding(
            data.num_nodes, args.hidden_channels).to(device)
        input_dim += args.hidden_channels
    if use_features:
        print(data.x.shape[1])
        input_dim += data.x.shape[1]
    if args.feature_aug:
        input_dim += args.feature_aug_k
    gnn = GCN(
        input_dim, args.hidden_channels,
        args.hidden_channels, args.num_layers,
        args.dropout).to(device)
    linkpred = LinkPredictor(
        args.hidden_channels, args.hidden_channels,
        1, args.num_layers,
        args.dropout, aug).to(device)
    model = LinkGNN(emb, gnn, linkpred, aug)
    return model


def train(model, data, split_edge, optimizer, batch_size, use_params, device, dataset):
    model.train()
    pos_train_edge = split_edge['train']['edge'].to(device)

    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    total_loss = total_examples = 0
    running_loss = None
    alpha = 0.99
    running_acc = None

    for idx, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                                          shuffle=True)):
        if use_params:
            optimizer.zero_grad()
        pos_edge = to_undirected(pos_train_edge[perm].t(), data.num_nodes)

        # if dataset == "collab":
        neg_edge = torch.randint(0, data.num_nodes, pos_edge.size(
        ), dtype=torch.long, device=pos_edge.device)
        # else:
        #     neg_edge = negative_sampling(
        #         edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge.size(1), method='dense')

        out = model(data.x, torch.cat(
            [pos_edge, neg_edge], 1), data.adj_t).squeeze()

        pos_out = out[:pos_edge.size(1)]
        pos_loss = -torch.log(pos_out + 1e-8).mean()

        neg_out = out[pos_edge.size(1):]
        neg_loss = -torch.log(1 - neg_out + 1e-8).mean()

        loss = pos_loss + neg_loss

        if use_params:
            loss.backward()

        acc = ((neg_out < 0.5).sum() + (pos_out > 0.5).sum()
               ).item()/(0.+out.size(0))
        if running_loss is None:
            running_loss = loss.item()
            running_acc = acc
        running_loss = (1-alpha)*loss.item() + alpha*running_loss
        running_acc = (1-alpha)*acc + alpha*running_acc

        if use_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if use_params:
            optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def main():
    parser = argparse.ArgumentParser()
    # experiment configs
    parser.add_argument('--dataset', type=str, required=True)
    # options: ["gcn", "aa", "katz"]
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--runs', type=int, default=5)

    # model configs
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16*1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    # other settings
    parser.add_argument('--eval_steps', type=int, default=1)
    # file to which results will be appended
    parser.add_argument('--results', type=str, default=None)

    # feature augmentation settings
    parser.add_argument('--feature-aug', dest='feature_aug',
                        action='store_true')
    # the pairwise score metric to augment features with. Options: ['katz']
    parser.add_argument('--feature-aug-score', type=str, default='katz')
    # the number of k important nodes to augment features with
    parser.add_argument('--feature-aug-k', type=int, default=50)
    # percentage of random nodes to pick for augmentation
    parser.add_argument('--random-pct', type=float, default=0.0)
    parser.add_argument('--katz-beta', type=float, default=0.05)

    # embedding augmentation settings
    # the pairwise score metric to augment node embeddings with. Options: ['aa', 'cn']
    # If not supplied, defaults to None
    parser.add_argument('--embedding-aug', type=str, default=None)

    parser.set_defaults(feature_aug=False)
    args = parser.parse_args()

    print(args)

    best_hits = {
        "collab": "Hits@50",
        "ddi": "Hits@20",
        "block": "Hits@20"
    }

    edge_index, edge_weight, split_edge, data = load_data(args.dataset)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    data = data.to(device)

    def add_edges(edge_index, edge_weight, extra_edges, num_nodes):
        full_edge_index = torch.cat([edge_index.clone(), extra_edges], dim=-1)
        new_edge_weight = torch.ones(extra_edges.shape[1])
        full_edge_weights = torch.cat([edge_weight, new_edge_weight], 0)
        adj_t = SparseTensor.from_edge_index(
            full_edge_index, full_edge_weights, sparse_sizes=[num_nodes, num_nodes])
        adj_t = adj_t.to_symmetric()
        return adj_t

    adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=[
                                         data.num_nodes, data.num_nodes])
    adj_t = adj_t.to_symmetric()
    data.adj_t = adj_t.to(device).float()
    val_edge_index = split_edge['valid']['edge'].t()
    print("Line 257, models.py")
    print("val edge index shape")
    print(val_edge_index.size())
    print("edge index shape")
    print(edge_index.size())
    val_edge_index = to_undirected(val_edge_index)
    data.full_adj_t = add_edges(
        edge_index, edge_weight, val_edge_index, data.num_nodes).to(device).float()

    # TODO: might have to change if we start using non ogb datasets
    if args.dataset == 'block':
        evaluator = Evaluator('ogbl-ddi')
    else:
        evaluator = Evaluator(name='ogbl-' + args.dataset)

    if args.model == "gcn":
        if args.feature_aug:
            print("Augmenting features")
            if args.feature_aug_score == "katz":
                augmented_tr, augmented_te = pagerank_augment(
                    data, args.dataset, args.feature_aug_k, args.katz_beta, args.random_pct)
        use_features = True if args.dataset == "collab" else False
        model = build_gcn(args, data, device, use_features,
                          True, aug=args.embedding_aug)

        loggers = {}
        for K in hits[args.dataset]:
            loggers[f"Hits@{K}"] = Logger(args.runs)

        for run in range(args.runs):
            model.reset_parameters()
            use_params = sum(p.numel()
                             for p in model.parameters() if p.requires_grad) > 0
            print(sum(p.numel()
                      for p in model.parameters() if p.requires_grad))
            if use_params:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            else:
                optimizer = None

            if not use_params:
                args.epochs = 1

            highest_eval = 0
            for epoch in tqdm(range(1, 1 + args.epochs)):
                if use_params:
                    if args.feature_aug:
                        data.x = augmented_tr
                    loss = train(model, data, split_edge, optimizer,
                                 args.batch_size, use_params, device, args.dataset)
                else:
                    loss = -1

                if epoch % args.eval_steps == 0:
                    if args.feature_aug:
                        data.x = augmented_te
                    results = test_gcn(model, data, split_edge, evaluator,
                                       args.batch_size, device, args.dataset)
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                    if epoch % 2 == 0:
                        for key, result in results.items():
                            train_hits, valid_hits, test_hits = result
                            if key == best_hits[args.dataset]:
                                if valid_hits >= highest_eval:
                                    highest_eval = valid_hits

                            print(key)
                            print(f'Run: {run + 1:02d}, '
                                  f'Epoch: {epoch:02d}, '
                                  f'Loss: {loss:.4f}, '
                                  f'Train: {100 * train_hits:.2f}%, '
                                  f'Valid: {100 * valid_hits:.2f}%, '
                                  f'Test: {100 * test_hits:.2f}%')
                        print('---')

            for key in loggers.keys():
                print(key)
                loggers[key].print_statistics(run)

                if key == best_hits[args.dataset]:
                    result = 100 * torch.tensor(loggers[key].results[run])
                    argmax = result[:, 1].argmax().item()
                    curve_point = [result[argmax, 1], result[argmax, 2]]

            print(curve_point)

        for key in loggers.keys():
            print(key)
            final_ans = loggers[key].print_statistics()
            if key == best_hits[args.dataset] and args.results:
                with open(args.results, "a") as f:
                    params = []
                    if args.feature_aug:
                        params.append(args.feature_aug_score)
                        params.append(args.feature_aug_k)
                        params.append(args.random_pct)
                        params.append(args.katz_beta)
                    if args.embedding_aug:
                        params.append(args.embedding_aug)
                    params = [str(i) for i in params]
                    f.write(", ".join(params + final_ans))
                    f.write("\n")
    elif args.model == "aa":
        results = test_adamic(data, split_edge, evaluator,
                              device, args.dataset)
        print(results)
    elif args.model == "katz":
        results = test_katz(data, split_edge, evaluator,
                            args.batch_size, device, args.dataset)
        print(results)


if __name__ == "__main__":
    main()
