import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric.utils import coalesce
import torch_geometric as pyg

from model.layers import LinkDecoder, MPNN, MPNN2, TimeEncode, Sampling

class DyGCN(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, n_layers, num_heads, window_size,
                 ratio=0.8, time_encode=True, recurrent=True, device='cuda'):
        super(DyGCN, self).__init__()
        
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.ratio = ratio
        self.n_layers = n_layers
        self.time_encode = time_encode
        self.recurrent = recurrent
        self.num_heads = num_heads
        
        self.device = device

        self.mlp_transform = nn.Sequential(nn.Linear(self.dim_in, self.hidden_dim),
                                           nn.ReLU())
        
        if self.time_encode:
            self.edge_time_encode = TimeEncode(self.hidden_dim)
            self.sampling_input_dim = self.hidden_dim * 3
        else:
            self.sampling_input_dim = self.hidden_dim * 2
            
        if self.window_size != 1:
            self.sampling = Sampling(self.sampling_input_dim, self.hidden_dim)
            
        self.memory_edge_index = []
        self.memory_edge_time = []
        self.memory_embedding = []

        self.layer = nn.ModuleList([pyg.nn.conv.GCNConv(self.hidden_dim, self.hidden_dim)
                                    for i in range(self.n_layers)])
        self.act = nn.ReLU()
        
        if self.recurrent:
            self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)

    def update_memory(self, edge_index, edge_time):
        if len(self.memory_edge_index) >= self.window_size:
            self.memory_edge_index.pop(0)
            self.memory_edge_time.pop(0)

        self.memory_edge_index.append(edge_index.clone().detach().cpu())
        self.memory_edge_time.append(edge_time.clone().detach().cpu())

    def reset_memory(self):
        self.memory_edge_index = []
        self.memory_edge_time = []

    def read_memory(self, edge_index, edge_time):
        self.memory_edge_index = edge_index
        self.memory_edge_time = edge_time

    def merge_graphs(self, edge_index_list, edge_time_list):
        merge = torch.cat(edge_index_list, dim=-1)
        merge_time = torch.cat(edge_time_list, dim=-1)
        merge, merge_time = coalesce(merge, merge_time, reduce='max')
        return merge, merge_time

    def merge_memory(self):
        merge_graph, merge_time = self.merge_graphs(self.memory_edge_index, self.memory_edge_time)
        return merge_graph.to(self.device), merge_time.to(self.device)

    def forward(self, x, edge_index, edge_label_index, edge_feature, previous_state, is_updated=False):

        x = self.mlp_transform(x)

        if not is_updated and self.window_size != 1:
          self.update_memory(edge_index, edge_feature)

        if self.window_size != 1:
            merge_edge_index, merge_edge_feature = self.merge_memory()
            merge_edge_feature = (merge_edge_feature - merge_edge_feature.min()) + 1
            
            if self.time_encode:
                merge_edge_feature = self.edge_time_encode(merge_edge_feature)
                edge_logits = self.sampling(previous_state, merge_edge_index, merge_edge_feature)
            else:
                edge_logits = self.sampling(previous_state, merge_edge_index, None)

            if self.training:
                temperature = 1.0
                bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
                gate_inputs = torch.log(eps + 1e-15) - torch.log(1 - eps + 1e-15)
                gate_inputs = gate_inputs.to(self.device)
                gate_inputs = (gate_inputs + edge_logits) / temperature
            else:
                gate_inputs = edge_logits

            z = torch.sigmoid(gate_inputs).squeeze()
            __, sorted_idx = z.sort(dim=-1, descending=True)

            k = int(self.ratio * z.size(0))
            keep = sorted_idx[:k]
            sampling_edge_logits = torch.sigmoid(gate_inputs).squeeze()

            sampling_edge_weight = torch.gather(sampling_edge_logits, dim=0, index=keep)
            sampling_edge_index = torch.gather(merge_edge_index, dim=1, index=torch.stack([keep, keep], dim=0))
            
            if self.time_encode:
                sampling_edge_feature = torch.gather(merge_edge_feature, dim=0, index=keep.unsqueeze(dim=1).repeat(1, merge_edge_feature.shape[1]))
                for i, layer in enumerate(self.layer):
                    x = layer(x, sampling_edge_index, sampling_edge_weight)
                    if i != self.n_layers - 1:
                        x = self.act(x)
            else:
                for i, layer in enumerate(self.layer):
                    x = layer(x, sampling_edge_index, sampling_edge_weight)
                    if i != self.n_layers - 1:
                        x = self.act(x)
        else:
            for i, layer in enumerate(self.layer):
                    x = layer(x, edge_index)
                    if i != self.n_layers - 1:
                        x = self.act(x)
        
        if self.recurrent:
            x = self.gru(x, previous_state)
        prediction = self.decoder(x, edge_label_index)
        return prediction, x
    
class MSTAGNN(torch.nn.Module):
    def __init__(self, hidden_dim, dropout, K, num_heads):
        super(MSTAGNN, self).__init__()

        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.K = K
        self.num_heads = num_heads

        self.linQ = nn.Linear(self.hidden_dim, self.head_dim * num_heads)
        self.linK = nn.Linear(self.hidden_dim, self.head_dim * num_heads)
        self.linV = nn.Linear(self.hidden_dim, self.head_dim * num_heads)

        self.mpnnM = MPNN2(node_dim=0)
        self.mpnnK = MPNN(node_dim=0)

        self.hopwise = nn.Parameter(torch.ones(K + 1))

    def forward(self, x, edge_index, edge_feature, edge_weight):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * deg_inv[col] * edge_weight

        Q = self.linQ(x)
        K = self.linK(x)
        V = self.linV(x)

        Q = F.relu(Q)

        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        hidden = V * self.hopwise[0]

        if edge_feature is not None:
            edge_feature = edge_feature.view(-1, self.num_heads, self.head_dim)

        for hop in range(self.K):
            if hop == 0:
                M = self.mpnnM(K, V, edge_index, norm.view(-1, 1, 1, 1), edge_feature)
                K = self.mpnnK(K, edge_index, norm.view(-1, 1, 1), edge_feature, act=True)
            else:
                M = self.mpnnK(M, edge_index, norm.view(-1, 1, 1, 1))
                K = self.mpnnK(K, edge_index, norm.view(-1, 1, 1))

            H = torch.einsum('nhi,nhij->nhj', [Q, M])
            normalize = torch.norm(H / math.sqrt(self.head_dim), p=2, dim=-1)
            Xnorm = torch.maximum(torch.ones(normalize.shape).to(normalize.device), normalize)
            H = H / Xnorm.unsqueeze(-1)

            gamma = self.hopwise[hop + 1]

            hidden = hidden + gamma * H

        hidden = hidden.view(-1, self.head_dim * self.num_heads)
        return hidden