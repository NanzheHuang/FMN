import os
import json
import argparse
import random
import time

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim, Tensor

from data.md22.dataset import MD22Dataset
from model.backbone import EPPM

parser = argparse.ArgumentParser(description='EPPM')

# 1. Experiment
parser.add_argument('--exp_name', type=str, default='DHA_test250603_01', metavar='N', help='experiment_name')
parser.add_argument('--mol_name', type=str, default='DHA', help='Name of the molecule.')
parser.add_argument('--data_dir', type=str, default='data/md22', help='Data directory.')
parser.add_argument('--delta_frame', type=int, default=3000, help='Number of frames delta.')
parser.add_argument('--num_time_steps', type=int, default=8, help='Number of anchor to be predicted.')
parser.add_argument('--max_training_samples', type=int, default=500, help='maximum amount of training samples')
parser.add_argument('--max_val_samples', type=int, default=2000, help='maximum amount of validating samples')
parser.add_argument('--max_testing_samples', type=int, default=2000, help='maximum amount of testing samples')

# 2. Training
parser.add_argument('--model', type=str, default='EPPM', metavar='N', help='available models: EPPM')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50000, help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before logging test')
parser.add_argument('--out_f', type=str, default='res/res_md22', help='folder to output the json log file')
parser.add_argument('--max_trainloss', type=float, default=5e+9, help='Stop training if train loss over 200')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 5e-4 8e-4 1e-4
parser.add_argument('--weight_decay', type=float, default=1e-12, help='weight decay')  # 1e-10

# 3. Net Control
parser.add_argument('--interaction_layer', type=int, default=4, help='The number of interaction layers per block.')
parser.add_argument('--channel', type=int, default=4, help='channel num in eq pipe')

args = parser.parse_args()
if args.mol_name == 'AT-AT-CG-CG' or args.mol_name == 'buckyball-catcher' or args.mol_name == 'double-walled_nanotube':
    max_training_samples = 200
    max_val_samples = 250
    max_testing_samples = 250
else:
    max_training_samples = args.max_training_samples
    max_val_samples = args.max_val_samples
    max_testing_samples = args.max_testing_samples

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:3" if args.cuda else "cpu")
print(device)
loss_mse = nn.MSELoss()

mol_select = {
    'Ac-Ala3-NHMe': [20, 88],
    'DHA': [24, 92],
    'stachyose': [45, 204],
    'AT-AT': [38, 188],
    'AT-AT-CG-CG': [76, 376],               # 500/1186/1186
    'buckyball-catcher': [120, 940],         # 500/473/473
    'double-walled_nanotube': [326, 2714],    # 500/285/285
}

dic = {
    'data': {
        'batch_size': args.batch_size,
        'channel_size': args.channel,
        'node_num': mol_select[args.mol_name][0],
        'edge_num': mol_select[args.mol_name][1],
        'node_attr_size': 2,  # value will also change [emb][inv][out_dim]
        'edge_attr_size': 5,
        'device': device, },
    'emb': {
        'k': 4,             # select top k from eigen vector
        'each_dim': 64,
        'dropout': 0},
    'eq': {
        'hid_dim': 64,
        'act': 'SiLU'},
    'pred': {
        'out_dim': 128,  # 256
        'act': 'SiLU',
        'head_num': 2,
        'dropout': 0.2,
        'use_bias': True,
        'layer_num': args.interaction_layer}
}

try:
    os.makedirs(args.out_f)
except OSError:
    pass
try:
    os.makedirs(args.out_f)
except OSError:
    pass

Time_list = []

def main():
    # Part1: Seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(args.exp_name)

    # Part2: Load Data
    dataset_train = MD22Dataset(partition='train', max_samples=max_training_samples, data_dir=args.data_dir,
                                molecule_type=args.mol_name, delta_frame=args.delta_frame)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)
    dataset_val = MD22Dataset(partition='val', max_samples=max_val_samples, data_dir=args.data_dir,
                              molecule_type=args.mol_name, delta_frame=args.delta_frame)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=8)
    dataset_test = MD22Dataset(partition='test', max_samples=max_testing_samples, data_dir=args.data_dir,
                               molecule_type=args.mol_name, delta_frame=args.delta_frame)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    # Part3: Load Model
    if args.model == 'EPPM':
        model = EPPM(dic)
    else:
        raise Exception("Wrong model specified")

    param_count = sum(torch.numel(param) for param in model.parameters())
    print(f"模型总参数量: {param_count}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Part4: Train
    results = {'epochs': [], 'loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
            print("\n*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))

        json_object = json.dumps(results, indent=4)
        with open(args.out_f + "/" + args.exp_name + ".json", "w") as outfile:
            outfile.write(json_object)

        if train_loss > args.max_trainloss:
            break

    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0, 'loss_vel': 0, 'reg_loss': 0}
    for batch_idx, data in enumerate(loader):
        B, N, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        loc, vel, edge_attr, charges, loc_end, vel_end, Z = data
        loc = loc.detach()
        edges = loader.dataset.get_edges(B, N)
        edges = [edges[0].to(device), edges[1].to(device)]

        optimizer.zero_grad()
        reg_loss = 0  # helper to compute reg loss

        if args.model == 'EPPM':
            if backprop == False:
                st = time.time()

            node = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            node = torch.cat((node, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, dim=1, keepdim=True)  # relative distances
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(loc, node, edges, edge_attr, vel)

            if backprop == False:
                end = time.time()
                # print((end-st)/args.batch_size*1e5)
                Time_list.append((end-st)/args.batch_size*1e5)

            if epoch == 10:
                print("Time: %.2f" % torch.tensor(Time_list).mean(dim=0, keepdim=True))

        else:
            raise Exception("Wrong model")

        loss = loss_mse(loc_pred, loc_end)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item() * B
        try:
            res['reg_loss'] += reg_loss.item() * B
        except:  # no reg loss (no sticks and hinges)
            pass
        res['counter'] += B

    if not backprop:
        prefix = "\n==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f reg loss: %.5f'
          % (prefix + loader.dataset.partition,
             epoch,
             res['loss'] / res['counter'],
             res['reg_loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == '__main__':
    if args.cuda:
        print("\n---------- CUDA is Running ----------\n")
    else:
        print("\n---------- CPU is Running ----------\n")
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
