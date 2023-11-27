import os, shutil, wget
import gzip
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import math

import torch

from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset

from scipy import sparse as sp
from torch_geometric.utils import coalesce

class UCIDataset():
    def __init__(self, url, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.url = url
        self.file_name = file_name
        self.time_interval = interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.download()
        self.preprocess()

        if self.time_interval in ['D', 'W', 'M']:
          self.make_graph_snapshot()
        else:
          self.split_by_seconds(self.time_interval)

        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = GraphDataset(
            self.snapshot_list,
            task='link_pred',
            edge_negative_sampling_ratio=1,
            minimum_node_per_graph=5
        )

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))
    
    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end

    def download(self):
        if not os.path.exists('./data/{}'.format(self.file_name)):
            if not os.path.exists('./data'):
                os.mkdir('./data')
            print('Downloading ...')
            wget.download(self.url, out='./data')
            print('Done')
        else:
            print("File Already Downloaded")

    def preprocess(self):
        path = './data/{}'.format(self.file_name)
        with gzip.open(path, 'rb') as f_in:
            with open('./data/CollegeMsg.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        path = './data/{}'.format('CollegeMsg.txt')
        df_trans = pd.read_csv(path, sep=' ', header=None)
        df_trans.columns = ['SRC', 'DST', 'TIMESTAMP']
        df_trans.reset_index(drop=True, inplace=True)

        # Node IDs of this dataset start from 1.
        df_trans['SRC'] -= 1
        df_trans['DST'] -= 1

        edge_index = torch.Tensor(df_trans[['SRC', 'DST']].values.transpose()).long()
        num_nodes = torch.max(edge_index) + 1

        node_feature = torch.rand(num_nodes, 64)

        node_id = torch.arange(num_nodes)

        edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

        self.graph = Graph(node_feature=node_feature,
                           edge_index=edge_index,
                           edge_time=edge_time)

    def split_by_seconds(self, freq_sec):
        split_criterion = self.graph.edge_time // freq_sec
        groups = torch.sort(torch.unique(split_criterion))[0]
        self.snapshot_list = list()
        for t in groups:
            period_members = (split_criterion == t)
            g = Graph(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members]
                      )
            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] > 2:
              self.snapshot_list.append(g)

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Graph(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            # positional_embedding = laplacian_positional_encoding(g, pos_enc_dim=8)
            g.positional_embedding = torch.zeros(1)

            if g.edge_index.shape[1] > 2:
              g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
              t = t + 1
              self.snapshot_list.append(g)

class MathOFDataset():
    def __init__(self, url, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.url = url
        self.file_name = file_name
        self.time_interval = interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.download()
        self.preprocess()

        if self.time_interval in ['D', 'W', 'M']:
          self.make_graph_snapshot()
        else:
          self.split_by_seconds(self.time_interval)

        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = GraphDataset(
            self.snapshot_list,
            task='link_pred',
            edge_negative_sampling_ratio=1,
            minimum_node_per_graph=5
        )

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))
    
    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end

    def download(self):
        if not os.path.exists('./data/{}'.format(self.file_name)):
            if not os.path.exists('./data'):
                os.mkdir('./data')
            print('Downloading ...')
            wget.download(self.url, out='./data')
            print('Done')
        else:
            print("File Already Downloaded")

    def preprocess(self):
        path = './data/{}'.format(self.file_name)
        with gzip.open(path, 'rb') as f_in:
            with open('./data/sx-mathoverflow.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        path = './data/{}'.format('sx-mathoverflow.txt')
        df_trans = pd.read_csv(path, sep=' ', header=None)
        df_trans.columns = ['SOURCE', 'TARGET', 'TIMESTAMP']
        df_trans.reset_index(drop=True, inplace=True)

        node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
        enc = OrdinalEncoder(categories=[node_indices, node_indices])
        raw_edges = df_trans[['SOURCE', 'TARGET']].values
        edge_index = enc.fit_transform(raw_edges).transpose()
        edge_index = torch.LongTensor(edge_index)

        num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

        node_feature = torch.rand(num_nodes, 64)

        node_id = torch.arange(num_nodes)

        edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

        self.graph = Graph(node_feature=node_feature,
                           edge_index=edge_index,
                           edge_time=edge_time)

    def split_by_seconds(self, freq_sec):
        split_criterion = self.graph.edge_time // freq_sec
        groups = torch.sort(torch.unique(split_criterion))[0]
        self.snapshot_list = list()
        for t in groups:
            period_members = (split_criterion == t)
            g = Graph(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members]
                      )
            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] > 2:
              self.snapshot_list.append(g)

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Graph(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            # positional_embedding = laplacian_positional_encoding(g, pos_enc_dim=8)
            g.positional_embedding = torch.zeros(1)

            if g.edge_index.shape[1] > 2:
              g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
              t = t + 1
              self.snapshot_list.append(g)

class BitCoinDataset():
    def __init__(self, url, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.url = url
        self.file_name = file_name
        self.time_interval = interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.download()
        self.preprocess()

        if self.time_interval in ['D', 'W', 'M']:
          self.make_graph_snapshot()
        else:
          self.split_by_seconds(self.time_interval)

        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = GraphDataset(
            self.snapshot_list,
            task='link_pred',
            edge_negative_sampling_ratio=1,
            minimum_node_per_graph=5
        )

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))

    def download(self):
        if not os.path.exists('./data/{}'.format(self.file_name)):
            if not os.path.exists('./data'):
                os.mkdir('./data')
            print('Downloading ...')
            wget.download(self.url, out='./data')
            print('Done')
        else:
            print("File Already Downloaded")

    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end
    
    def preprocess(self):
        path = './data/{}'.format(self.file_name)
        with gzip.open(path, 'rb') as f_in:
            with open('./data/soc-sign-bitcoinotc.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        path = './data/{}'.format('soc-sign-bitcoinotc.csv')
        df_trans = pd.read_csv(path, sep=',', header=None, index_col=None)

        df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']

        node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
        enc = OrdinalEncoder(categories=[node_indices, node_indices])
        raw_edges = df_trans[['SOURCE', 'TARGET']].values
        edge_index = enc.fit_transform(raw_edges).transpose()
        edge_index = torch.LongTensor(edge_index)

        num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

        node_feature = torch.rand(num_nodes, 64)

        edge_time = torch.FloatTensor(df_trans['TIME'].values)

        self.graph = Graph(node_feature=node_feature,
                           edge_index=edge_index,
                           edge_time=edge_time)

    def split_by_seconds(self, freq_sec):
        split_criterion = self.graph.edge_time // freq_sec
        groups = torch.sort(torch.unique(split_criterion))[0]
        self.snapshot_list = list()
        for t in groups:
            period_members = (split_criterion == t)
            g = Graph(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      edge_time=self.graph.edge_time[period_members])
            if g.edge_index.shape[1] >= 10:
              self.snapshot_list.append(g)

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Graph(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            # positional_embedding = laplacian_positional_encoding(g, pos_enc_dim=3)

            g.positional_embedding = torch.zeros(1)

            if g.edge_index.shape[1] >= 10:
              g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
              t = t + 1
              self.snapshot_list.append(g)

class DBLPDataset():
    def __init__(self, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.file_name = file_name
        self.time_interval = interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.preprocess()

        self.make_graph_snapshot()
        
        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = GraphDataset(
            self.snapshot_list,
            task='link_pred',
            edge_negative_sampling_ratio=1,
            minimum_node_per_graph=5
        )

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))

    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end
    
    def preprocess(self):
        path = './data/{}'.format(self.file_name)
        df = pd.read_csv(path)
        df['u'] -= 1
        df['i'] -= 1
        
        edge_index = torch.Tensor(df[['u', 'i']].values.transpose()).long()  # (2, E)
        num_nodes = torch.max(edge_index) + 1

        node_feature = torch.rand(num_nodes, 64)
        edge_time = torch.FloatTensor(df['ts'].values)

        self.graph = Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            edge_time=edge_time,
        )

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Graph(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            # positional_embedding = laplacian_positional_encoding(g, pos_enc_dim=3)
            g.positional_embedding = torch.zeros(1)
            g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
            t = t + 1
            self.snapshot_list.append(g)