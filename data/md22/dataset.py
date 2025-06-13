import numpy as np
import torch
import pickle as pkl
import os
import networkx as nx
from networkx.algorithms import tree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MD22Dataset():
    def __init__(self, partition, max_samples, delta_frame, data_dir, molecule_type):
        # setup a split, tentative setting
        train_par, val_par, test_par = 0.4, 0.2, 0.2
        full_dir = os.path.join(data_dir, molecule_type + '.npz')
        split_dir = os.path.join(data_dir, molecule_type + '_split.pkl')
        data = np.load(full_dir)
        self.partition = partition
        self.molecule_type = molecule_type

        x = data['R']
        v = x[1:] - x[:-1]
        x = x[:-1]

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except:
            get_split(x, delta_frame, train_par, test_par, val_par, split_dir)

        if partition == 'train':
            st = split[0]
        elif partition == 'val':
            st = split[1]
        elif partition == 'test':
            st = split[2]
        else:
            raise NotImplementedError()

        st = st[:max_samples]

        z = data['z']
        x = x[:, z > 1, ...]
        v = v[:, z > 1, ...]
        z = z[z > 1]

        x_0, v_0 = x[st], v[st]
        x_t, v_t = x[st + delta_frame], v[st + delta_frame]

        print('Got {:d} samples!'.format(x_0.shape[0]))

        mole_idx = z
        n_node = mole_idx.shape[0]
        self.n_node = n_node

        _lambda = 1.6

        def d(_i, _j, _t):
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        n = z.shape[0]

        self.Z = torch.Tensor(z)

        atom_edges = torch.zeros(n, n).int()
        for i in range(n):
            for j in range(n):
                if i != j:
                    _d = d(i, j, 0)
                    if _d < _lambda:
                        atom_edges[i][j] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_node):
            for j in range(n_node):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 1])
                        assert not self.atom_edge2[i][j]
                    if self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 2])
                        assert not self.atom_edge[i][j]

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = edges  # [2, edge]

        all_edges = {}

        for i in range(n):
            for j in range(i + 1, n):
                _d = d(i, j, 0)
                if _d < _lambda:
                    idx_i, idx_j = z[i], z[j]
                    if idx_i < idx_j:
                        idx_i, idx_j = idx_j, idx_i
                    if (idx_i, idx_j) in all_edges:
                        all_edges[(idx_i, idx_j)].append([i, j])
                    else:
                        all_edges[(idx_i, idx_j)] = [[i, j]]

        # print(all_edges)
        # select the type of bonds to preserve the bond constraint
        conf_edges = []
        for key in all_edges:
            # if True:
            assert abs(key[0] - key[1]) <= 2
            conf_edges.extend(all_edges[key])

        # print(conf_edges)
        self.conf_edges = conf_edges
        self.x_0, self.v_0, self.x_t, self.v_t = torch.Tensor(x_0), torch.Tensor(v_0), torch.Tensor(x_t), torch.Tensor(
            v_t)
        self.mole_idx = torch.Tensor(mole_idx)

        # self.cfg = self.sample_cfg()

    def __getitem__(self, i):
        edge_attr = self.edge_attr
        edges = self.edges
        return (self.x_0[i], self.v_0[i], edge_attr, self.mole_idx.unsqueeze(-1),
                self.x_t[i], self.v_t[i], self.Z.unsqueeze(-1))

    def __len__(self):
        return len(self.x_0)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


def get_split(x, delta_frame, train_par, test_par, val_par, split_dir,
              rd_seed=100, start_per=0.1, end_per=0.02):
    """
    :param x:           [frame_num, N, 3]
    :param delta_frame: from t=0 to t=T, tT-t0 is delta_frame
    :param rd_seed:
    :param start_per:   Remove the first p percent of the data set
    :param end_per:     Remove the lase p percent of the data set
    :return:
    """
    np.random.seed(rd_seed)
    num = x.shape[0]
    start_num = int(num * start_per)
    end_num = int(num * (1 - end_per) - delta_frame)
    _x = x[start_num:end_num]

    train_idx = np.random.choice(np.arange(_x.shape[0]), size=int(train_par * _x.shape[0]), replace=False)
    flag = np.zeros(_x.shape[0])
    for _ in train_idx:
        flag[_] = 1
    rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
    val_idx = np.random.choice(rest, size=int(val_par * _x.shape[0]), replace=False)
    for _ in val_idx:
        flag[_] = 1
    rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
    test_idx = np.random.choice(rest, size=int(test_par * _x.shape[0]), replace=False)

    train_idx += start_num
    val_idx += start_num
    test_idx += start_num

    split = (train_idx, val_idx, test_idx)

    with open(split_dir, 'wb') as f:
        pkl.dump(split, f)
    print('Generate and save split!')


if __name__ == '__main__':
    dataset_train = MD22Dataset(partition='test', max_samples=2000, data_dir='',
                                molecule_type='double-walled_nanotube', delta_frame=3000)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=True, drop_last=True,
                                               num_workers=8)

    loader = loader_train

    for batch_idx, data in enumerate(loader):
        B, N, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        loc, vel, edge_attr, charges, loc_end, vel_end, Z = data
        loc = loc.detach()
        edges = loader.dataset.get_edges(B, N)
        edges = [edges[0].to(device), edges[1].to(device)]






