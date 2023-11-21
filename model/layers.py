import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric_temporal as pygt
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import zeros
    
class LinkDecoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LinkDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2 * dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, x, edge_label_index):
        x_i = x[edge_label_index[0]]
        x_j = x[edge_label_index[1]]
        prediction = self.mlp(torch.cat([x_i, x_j], dim=-1))
        return prediction
    
class MPNN(MessagePassing):
    def __init__(self, node_dim):
        super().__init__(aggr='add', node_dim=node_dim)
        
    def forward(self, x, edge_index, norm, edge_feature=None):
        out = self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_feature)
        return out

    def message(self, x_j, norm, edge_feature):
        if edge_feature is None:
            x_j = x_j
        else:
            x_j = x_j + edge_feature
        return norm * x_j

# SubTree Linear Attention
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

        self.output_layer = nn.Linear(self.head_dim * num_heads, self.hidden_dim)

        self.mpnnM = MPNN(node_dim=-4)
        self.mpnnK = MPNN(node_dim=-3)

        self.cst = 10e-6

        self.hopwise = nn.Parameter(torch.ones(K + 1))
        self.headwise = nn.Parameter(torch.zeros(size=(self.num_heads, K)))

    def forward(self, x, edge_index, edge_feature):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * deg_inv[col]

        Q = self.linQ(x)
        K = self.linK(x)
        V = self.linV(x)

        Q = 1 + F.elu(Q)
        K = 1 + F.elu(K)

        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        M = torch.einsum('nhi,nhj->nhij', [K, V])

        hidden = V * (self.hopwise[0])
        layerwise = F.softmax(self.headwise, dim=-2)

        for hop in range(self.K):

            M = self.mpnnM(M, edge_index, norm.view(-1, 1, 1, 1))
            K = self.mpnnK(K, edge_index, norm.view(-1, 1, 1))

            H = torch.einsum('nhi,nhij->nhj', [Q, M])
            C = torch.einsum('nhi,nhi->nh', [Q, K]).unsqueeze(-1) + self.cst
            H = H / C
            gamma = self.hopwise[hop + 1] * layerwise[:, hop].unsqueeze(-1)

            hidden = hidden + gamma * H

        hidden = hidden.view(-1, self.head_dim * self.num_heads)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.output_layer(hidden)

        return hidden
    
# SubTree without Attention
class STGNN(torch.nn.Module):
    def __init__(self, hidden_dim, dropout, K, num_heads):
        super(STGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.K = K

        # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mpnnM = MPNN(node_dim=0)

        self.hopwise = nn.Parameter(torch.ones(K + 1))

    def forward(self, x, edge_index, edge_feature):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * deg_inv[col]

        hidden = x * self.hopwise[0]
        
        for hop in range(self.K):
            x = self.mpnnM(x, edge_index, norm.view(-1, 1), edge_feature)
            gamma = self.hopwise[hop + 1]
            hidden = hidden + gamma * x

        return hidden
    
# SubTree Softmax Attention
class STSMGNN(nn.Module):
    def __init__(self, hidden_dim, dropout, K, num_heads):
        super(STSMGNN, self).__init__()
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.K = K
        self.num_heads = num_heads

        self.linQ = nn.Linear(self.hidden_dim, self.head_dim * num_heads)
        self.linK = nn.Linear(self.hidden_dim, self.head_dim * num_heads)
        self.linV = nn.Linear(self.hidden_dim, self.head_dim * num_heads)

        self.mpnn = MPNN(node_dim=1)

        self.output_layer = nn.Linear(self.head_dim * num_heads, self.hidden_dim)

        self.hopwise = nn.Parameter(torch.ones(K + 1))
        self.headwise = nn.Parameter(torch.zeros(size=(self.num_heads, K)))

    def forward(self, x, edge_index, edge_feature):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * deg_inv[col]

        Q = self.linQ(x)
        K = self.linK(x)
        V = self.linV(x)

        Q = Q.view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = K.view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = V.view(-1, self.num_heads, self.head_dim)

        hidden = V * (self.hopwise[0])
        layerwise = F.softmax(self.headwise, dim=-2)

        # H N D x H D N -> H N N
        sim = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
        sim = torch.exp(sim)
        
        for hop in range(self.K):
            
            sim = self.mpnn(sim, edge_index, norm.view(1, -1, 1))
            
            H = torch.matmul(sim, V.transpose(1, 0))
            H = H.permute(1, 0, 2)
            C = torch.sum(sim.permute(1, 0, 2), dim=-1).unsqueeze(-1) + 10e-6
            H = H / C

            gamma = self.hopwise[hop + 1] * layerwise[:, hop].unsqueeze(-1)
            hidden = hidden + gamma * H

        hidden = hidden.view(-1, self.head_dim * self.num_heads)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.output_layer(hidden)
        return hidden

class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.0):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, t):
        return self.lin(self.emb(t))

class RolandConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(RolandConvLayer, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_msg = nn.Linear(self.in_channels * 2, self.out_channels, bias=False)
        self.linear_skip = nn.Linear(self.in_channels, self.out_channels, bias=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        skip_x = self.linear_skip(x)

        return self.propagate(edge_index, x=x, norm=None) + skip_x

    def message(self, x_i, x_j, norm):
        x_j = torch.cat((x_i, x_j), dim=-1)
        x_j = self.linear_msg(x_j)
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
            
class GRUUpdater(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRUUpdater, self).__init__()

        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())

        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())

        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())

    def forward(self, x, H):
        H_prev = H
        X = x
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde
        H_out = H_gru
        return H_out

class RolandLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(RolandLayer, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.layer = pyg.nn.GCNConv(dim_in, dim_out)

        self.post_layer = nn.Sequential(nn.BatchNorm1d(dim_out),
                                        nn.PReLU())
        
        self.embedding_updater = GRUUpdater(dim_in, dim_out)

    def _init_hidden_state(self, x, H):
        if not isinstance(H, torch.Tensor):
            H = torch.zeros(x.shape[0], self.dim_out)
        return H.to(x.device)
    
    def forward(self, x, edge_index, H):
        H = self._init_hidden_state(x, H)
        x = self.layer(x, edge_index)
        x = self.post_layer(x)
        x = self.embedding_updater(x, H)
        return x

class EGCNLayer(nn.Module):
    def __init__(self, hidden_dim, num_nodes, dropout=0.0, rnn='LSTM'):
        super(EGCNLayer, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if rnn == 'LSTM':
            self.layer = pygt.EvolveGCNO(self.hidden_dim)
        elif rnn == 'GRU':
            self.layer = pygt.EvolveGCNH(num_nodes, self.hidden_dim)
        else:
            raise NotImplementedError('No such RNN model {}'.format(rnn))

        self.post_layer = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(self.dropout))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.layer(x, edge_index, edge_weight)
        x = self.post_layer(x)
        return x