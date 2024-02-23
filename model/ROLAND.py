import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import zeros

from model.layers import LinkDecoder

class ROLAND(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, num_layer):
        super(ROLAND, self).__init__()

        self.num_layer = num_layer
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.mlp_transform = nn.Sequential(nn.Linear(self.dim_in, self.hidden_dim),
                                           nn.BatchNorm1d(self.hidden_dim),
                                           nn.ReLU())

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

        self.layer = RolandConvLayer(dim_in, dim_out)

        self.post_layer = nn.Sequential(nn.BatchNorm1d(dim_out),
                                        nn.ReLU())

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