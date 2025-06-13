import torch
import torch.utils.data
from torch import nn, optim
import numpy as np
from data.motion.dataset import MotionDataset
import os
import argparse
import json
import csv
import time
from collections import deque
import statistics
import random

from model.backbone import EPPM


parser = argparse.ArgumentParser(description='EPPM')
# 1. Experiment
parser.add_argument('--exp_name', type=str, default='Table-4-woFA-walk', metavar='N', help='experiment_name')
parser.add_argument('--case', type=str, default='walk', help='Name of the molecule.')
parser.add_argument('--data_dir', type=str, default='data/motion/dataset', help='Data directory.')
parser.add_argument('--delta_frame', type=int, default=30, help='Number of frames delta.')
parser.add_argument('--max_training_samples', type=int, default=200, help='maximum amount of training samples')
parser.add_argument('--max_val_samples', type=int, default=600, help='maximum amount of validating samples')
parser.add_argument('--max_testing_samples', type=int, default=600, help='maximum amount of testing samples')

# 2. Training
parser.add_argument('--model', type=str, default='EPPM', metavar='N', help='available models: EPPM')
parser.add_argument('--batch_size', type=int, default=12, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before logging test')
parser.add_argument('--out_f', type=str, default='res/res_mocap', help='folder to output the json log file')
parser.add_argument('--max_trainloss', type=float, default=1000, help='Stop training if train loss over 200')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')     # 5e-4 8e-4 1e-4
parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay')   # 1e-10

# 3. Net Control
parser.add_argument('--interaction_layer', type=int, default=4, help='The number of interaction layers per block.')
parser.add_argument('--channel', type=int, default=1, help='channel num in eq pipe')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:1" if args.cuda else "cpu")
print(device)
loss_mse = nn.MSELoss()
loss_mse2 = nn.MSELoss()

try:
    os.makedirs(args.out_f)
except OSError:
    pass

dic = {
    'data': {
        'case': args.case,
        'batch_size': args.batch_size,
        'channel_size': args.channel,
        'node_num': 31,
        'edge_num': 130,
        'node_attr_size': 2,       # value will also change [emb][inv][out_dim]
        'edge_attr_size': 2,
        'device': device, },
    'emb': {
        'k': 4,             # select top k from eigen vector
        'each_dim': 64},
    'eq': {
        'hid_dim': 64,
        'act': 'SiLU'},
    'pred': {
        'hid_dim': 32,
        'out_dim': 256,     # 256
        'act': 'SiLU',
        'head_num': 2,
        'dropout': 0.2,
        'use_bias': True,
        'layer_num': args.interaction_layer}
}


def main():
    # Part1: Seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Part2: Load Data
    dataset_train = MotionDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame, case=args.case)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)
    dataset_val = MotionDataset(partition='val', max_samples=args.max_val_samples, data_dir=args.data_dir,
                                delta_frame=args.delta_frame, case=args.case)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=8)
    dataset_test = MotionDataset(partition='test', max_samples=args.max_testing_samples, data_dir=args.data_dir,
                                 delta_frame=args.delta_frame, case=args.case)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    # Part3: Load Model
    if args.model == 'EPPM':
        model = EPPM(dic)
    else:
        raise NotImplementedError('Unknown model:', args.model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)

            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d "
                  % (best_val_loss, best_test_loss, best_epoch))

        json_object = json.dumps(results, indent=4)
        with open(args.out_f + "/" + args.exp_name + ".json", "w") as outfile:
            outfile.write(json_object)
    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        loc, vel, edges, edge_attr, local_edges, local_edge_fea, Z, loc_end, vel_end = data
        loc = loc.detach()
        vel = vel.detach()
        edges = edges.reshape(args.batch_size, 2, -1)
        offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        edges = torch.cat(list(edges + offset), dim=-1)     # [2, BM]

        optimizer.zero_grad()
        if args.model == 'EPPM':
            node = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            node = torch.cat((node, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, dim=1, keepdim=True)  # relative distances
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(loc, node, edges, edge_attr, vel)
        else:
            raise Exception("Wrong model")

        loss = loss_mse(loc_pred, loc_end)

        if backprop:
            if args.model == 'EPPM':
                loss.backward()
            else:
                loss.backward()
            optimizer.step()

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch:%d ,avg_loss:%.5f'
          % (prefix + loader.dataset.partition, epoch, res['loss'] / res['counter']))
    return res['loss'] / res['counter']


if __name__ == "__main__":

    if args.cuda:
        print("\n-----------Cuda is running!-----------\n")
    else:
        print("\n-----------C P U-----------\n")
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)








"""import torch
import torch.utils.data
from torch import nn, optim
import numpy as np
from data.motion.dataset import MotionDataset
import os
import argparse
import json
import csv
import time
from collections import deque
import statistics
import random

from model.backbone import EPPM


parser = argparse.ArgumentParser(description='Equivariant Point Prediction Model')

# Model Related
parser.add_argument('--model', type=str, default='EPPM', metavar='N',
                    help='available models: EPPM')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--flat', action='store_true', default=False,
                    help='flat MLP')
parser.add_argument('--interaction_layer', type=int, default=7,
                    help='The number of interaction layers per block.')
parser.add_argument('--pos_encode_mod', type=str, default='eig',
                    help='adjacency matrix(adj) / laplacian matrix(lap) / eigenvectors matrix(eig)')
parser.add_argument('--channel', type=int, default=4,
                    help='channel num in eq pipe')

# Data Related (Related to function in ./data/md17/dataset)
parser.add_argument('--case', type=str, default='run',  # for mocap, is 'walk' or 'run'
                    help='The case, walk or run.')
parser.add_argument('--max_training_samples', type=int, default=200, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--max_val_samples', type=int, default=240, metavar='N',
                    help='maximum amount of validating samples')
parser.add_argument('--max_testing_samples', type=int, default=240, metavar='N',
                    help='maximum amount of testing samples')
parser.add_argument('--data_dir', type=str, default='data/motion/dataset',
                    help='Data directory.')
parser.add_argument('--delta_frame', type=int, default=30,
                    help='Number of frames delta.')

# Training Related
parser.add_argument('--exp_name', type=str, default='run', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='res/res_mocap', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=0.0005, metavar='N',
                    help='learning rate')  # 0.0005
parser.add_argument('--lr_Q', type=float, default=0.005,
                    help='the learning rate of matrix Q.')  # 0.005
parser.add_argument('--lr_Q_loss', type=float, default=0.005,
                    help='the learning rate of Q_loss.')
parser.add_argument('--lr_Q_loss_decay', type=float, default=10e-6,
                    help='the weight decay of Q_loss.')
parser.add_argument('--lr_mode', type=str, default='divided',
                    help='scheduler mode / regular mode / divided mode')
parser.add_argument('--weight_decay', type=float, default=1e-9, metavar='N',
                    help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save_Q_epoch', type=list, default=[0, 200, 400, 800, 1000],
                    help='In these epoch, save Q to local as csv file. [1st, 2nd]')
parser.add_argument('--Q_data_folder_name', type=str, default='Q',
                    help='The folder name that contain Q data.')
parser.add_argument("--config_by_file", default=False, action="store_true", )
args = parser.parse_args()

time_exp_dic = {'time': 0, 'counter': 0}
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:1" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

try:
    os.makedirs(args.outf)
except OSError:
    pass
try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

dic = {
    'data': {
        'batch_size': args.batch_size,
        'channel_size': args.channel,
        'node_num': 31,
        'edge_num': 130,
        'node_attr_size': 2,       # value will also change [emb][inv][out_dim]
        'edge_attr_size': 2,
        'device': device
    },
    'emb': {
        'k': 4,
        'each_dim': 32,
    },
    'eq': {
        'hid_dim': args.nf,
        'act': 'SiLU'
    },
    'pred': {
        'hid_dim': 32,
        'out_dim': 256,     # 256
        'act': 'SiLU',
        'head_num': 2,
        'use_bias': True,
        'layer_num': args.interaction_layer}
}



def main():
    # Part1: Seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    que50 = deque(maxlen=50)

    # Part2: Load Data
    dataset_train = MotionDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame, case=args.case)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)
    dataset_val = MotionDataset(partition='val', max_samples=args.max_val_samples, data_dir=args.data_dir,
                                delta_frame=args.delta_frame, case=args.case)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=8)
    dataset_test = MotionDataset(partition='test', max_samples=args.max_testing_samples, data_dir=args.data_dir,
                                 delta_frame=args.delta_frame, case=args.case)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    # Part3: Load Model
    if args.model == 'EPPM':
        model = EPPM(dic)
    else:
        raise NotImplementedError('Unknown model:', args.model)

    print("\nCalculating Positional Encoder for Transformer (No Need for Training)...")
    _, pos_encoder_matrix, _ = pos_encoder(loader_train, args.pos_encode_mod)
    pos_encoder_matrix = pos_encoder_matrix.to(device)
    print("Positional Encoder has been Initialized.\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_save_path = os.path.join(args.outf, args.exp_name, 'saved_model.pth')
    early_stopping = EarlyStopping(patience=50, verbose=True, path=model_save_path)

    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    bast_lp_loss = 1e8
    for epoch in range(0, args.epochs):
        if epoch == 0:
            start_time = time.time()
            train_loss, lp_loss = train(model, optimizer, epoch, loader_train, pos_encoder_matrix)
            end_time = time.time()
            print("Total time: " + str(end_time - start_time))
        else:
            train_loss, lp_loss = train(model, optimizer, epoch, loader_train, pos_encoder_matrix)

        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss, _ = train(model, optimizer, epoch, loader_val, pos_encoder_matrix, backprop=False)
            test_loss, _ = train(model, optimizer, epoch, loader_test, pos_encoder_matrix, backprop=False)

            que50.append(test_loss)
            que50_len = len(que50)

            mid, avg = calculate(que50)

            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                best_lp_loss = lp_loss
                # Save model is move to early stopping.
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d \t Mid: %.5f \t Avg: %.5f"
                  % (best_val_loss, best_test_loss, best_epoch, mid, avg))
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early Stopping.")
                break

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)
    return best_train_loss, best_val_loss, best_test_loss, best_epoch, best_lp_loss


def train(model, optimizer, epoch, loader, pos_en, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'Q_loss': 0}

    if epoch in args.save_Q_epoch:
        save_Q_flag = True
    else:
        save_Q_flag = False

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        # data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        loc, vel, edges, edge_attr, local_edges, local_edge_fea, Z, loc_end, vel_end = data
        # convert into graph minibatch
        loc = loc.view(-1, loc.size(2))
        vel = vel.view(-1, vel.size(2))
        offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        edges = torch.cat(list(edges + offset), dim=-1)  # [2, BM]
        edge_attr = torch.cat(list(edge_attr), dim=0)  # [BM, ]
        local_edge_index = torch.cat(list(local_edges + offset), dim=-1)  # [2, BM]
        local_edge_fea = torch.cat(list(local_edge_fea), dim=0)  # [BM, ]
        # local_edge_mask = torch.cat(list(local_edge_mask), dim=0)  # [BM, ]
        Z = Z.view(-1, Z.size(2))
        loc_end = loc_end.view(-1, loc_end.size(2))
        vel_end = vel_end.view(-1, vel_end.size(2))

        optimizer.zero_grad()

        if args.model == 'EPPM':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            node_attr = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred, vel_pred = model(loc, node_attr, edges, edge_attr, v=vel)
        else:
            raise Exception("Wrong model")

        loss = loss_mse(loc_pred, loc_end)

        if backprop:
            if args.model == 'EPPM':
                loss.backward()
            else:
                loss.backward()
            optimizer.step()

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch:%d ,avg_loss:%.5f'
          % (prefix + loader.dataset.partition, epoch, res['loss'] / res['counter']))
    return res['loss'] / res['counter'], res['Q_loss'] / res['counter']


def calculate(q):
    mid = statistics.median(q)
    avg = statistics.mean(q)
    return mid, avg


def pos_encoder(loader, mod='eigenvector'):
    # in:  data (loader); mod is to choose which value to return (adj? lap? eigenvector?)
    # out: encoded position
    # edge = (2, 1560) = B * 130
    # 12*31=372, so the graph node number will start from 0~371
    adj, laplacian, eigenvectors = None, None, None
    for batch_idx, data in enumerate(loader):
        B, N, _ = data[0].size()
        # Calculate Edge
        data = [d.to(device) for d in data]
        loc, vel, edges, edge_attr, local_edges, local_edge_fea, Z, loc_end, vel_end = data
        offset = (torch.arange(B) * N).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        edges = torch.cat(list(edges + offset), dim=-1)  # [2, BM]
        E = int(edges.size(-1) / B)
        # Get Adjacent Matrix (B*N, B*N)
        adj = torch.zeros([N, N], dtype=torch.float32)
        for i in range(E):
            adj[edges[0, i], edges[1, i]] = 1
        degree_matrix = torch.diag(torch.sum(adj, dim=1))
        laplacian = degree_matrix - adj
        eigenvalues, eigenvectors = torch.linalg.eig(laplacian)
        eigenvalues, eigenvectors = torch.real(eigenvalues), torch.real(eigenvectors)
        sorted_eigenvalues, indices = torch.sort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, indices]
        eigenvectors = eigenvectors[:, 0:4]
        break
    return adj, eigenvectors, laplacian


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, master_worker=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, master_worker)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, master_worker)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, master_worker=True):
        '''Saves model when validation loss decrease.'''
        if not master_worker:
            return
        if self.verbose and master_worker:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def get_counter(self):
        if (self.counter is not None) and (self.counter > 0):
            return self.counter
        else:
            return 0


if __name__ == "__main__":

    if args.cuda:
        print("\n-----------Cuda is running!-----------\n")
    else:
        print("\n-----------C P U-----------\n")

    ''' print new commers (para) '''
    '''print("lr: %f" % args.lr)
    print("lr_Q: %f" % args.lr_Q)
    print("interaction layers: %d" % args.interaction_layer)
    print("channel_num: %d" % args.channel)'''

    best_train_loss, best_val_loss, best_test_loss, best_epoch, best_lp_loss = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_lp = %.6f" % best_lp_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
    print("best_train = %.6f, best_lp = %.6f, best_val = %.6f, best_test = %.6f, best_epoch = %d"
          % (best_train_loss, best_lp_loss, best_val_loss, best_test_loss, best_epoch))

"""

