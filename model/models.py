import torch
import torch.nn as nn
import torch_geometric as pyg
from model.layers import *
from torch_geometric.utils import coalesce, dropout_edge

class ROLAND(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, num_layer):
        super(ROLAND, self).__init__()

        self.num_layer = num_layer
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.mlp_transform = nn.Sequential(nn.Linear(self.dim_in, self.hidden_dim),
                                           nn.BatchNorm1d(self.hidden_dim),
                                           nn.PReLU())
        
        self.layer = nn.ModuleList([RolandLayer(self.hidden_dim, self.hidden_dim)
                                    for i in range(self.num_layer)])

        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)

    def get_hidden_state(self, x, edge_index, H_list):
        hidden = []
        x = self.mlp_transform(x)
        for i, layer in enumerate(self.layer):
            x = layer(x, edge_index, H_list[i])
            hidden.append(x)

        return hidden

    def forward(self, x, edge_index, edge_label_index, H_list):
        x = self.mlp_transform(x)
        for i, layer in enumerate(self.layer):
            x = layer(x, edge_index, H_list[i])

        prediction = self.decoder(x, edge_label_index)
        return prediction

class EvolveGCN(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, num_layer, num_nodes, rnn='LSTM'):
        super(EvolveGCN, self).__init__()

        self.num_layer = num_layer
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.rnn = rnn

        self.layer = nn.ModuleList([EGCNLayer(self.hidden_dim, num_nodes, rnn=self.rnn)
                                    for i in range(self.num_layer)])

        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)

    def forward(self, x, edge_index, edge_label_index):
        for i, layer in enumerate(self.layer):
            x = layer(x, edge_index)

        prediction = self.decoder(x, edge_label_index)
        return prediction

def merge_graphs(edge_index_list, edge_time_list):
    merge = torch.cat(edge_index_list, dim=-1)
    merge_time = torch.cat(edge_time_list, dim=-1)
    merge, merge_time = coalesce(merge, merge_time, reduce='max')
    return merge, merge_time

class Model(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, num_layer, window_size):
        super(Model, self).__init__()

        self.num_layer = num_layer
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.window_size = window_size

        self.mlp_transform = nn.Sequential(nn.Linear(self.dim_in, self.hidden_dim),
                                           nn.ReLU())

        if self.window_size != 1:
            # self.edge_time_encode = RelTemporalEncoding(64, self.window_size)
            self.edge_time_encode = TimeEncode(self.hidden_dim)
            # self.edge_time_encode = GraphMixerTE(64)

        self.memory_edge_index = []
        self.memory_edge_time = []
        self.memory_embedding = []

        self.layer = STGNN(self.hidden_dim, dropout=0.0, K=5, num_heads=4)

        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)

    def update_memory(self, edge_index, edge_time):
        if len(self.memory_edge_index) >= self.window_size:
            self.memory_edge_index.pop(0)
            self.memory_edge_time.pop(0)

        self.memory_edge_index.append(edge_index.clone().detach())
        self.memory_edge_time.append(edge_time.clone().detach())

    def reset_memory(self):
        self.memory_edge_index = []
        self.memory_edge_time = []

    def read_memory(self, edge_index, edge_time):
        self.memory_edge_index = edge_index
        self.memory_edge_time = edge_time

    def merge_memory(self):
        merge_graph, merge_time = merge_graphs(self.memory_edge_index, self.memory_edge_time)
        return merge_graph, merge_time

    def forward(self, x, edge_index, edge_label_index, edge_feature, previous_state):

        x = self.mlp_transform(x)

        self.update_memory(edge_index, edge_feature)

        if self.window_size != 1:
            merge_edge_index, merge_edge_feature = self.merge_memory()
            merge_edge_feature =  (merge_edge_feature - merge_edge_feature.min() + 1)

            merge_edge_feature = self.edge_time_encode(merge_edge_feature)
            x = self.layer(x, merge_edge_index, merge_edge_feature)
        else:
            x = self.layer(x, edge_index, None)

        x = self.gru(x, previous_state)
        prediction = self.decoder(x, edge_label_index)
        return prediction, x