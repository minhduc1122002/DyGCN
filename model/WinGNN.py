import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

class WinGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super(WinGNNLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.transform = nn.Linear(in_channels, out_channels)

    def norm(self, edge_index, num_nodes, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        fill_value = 1
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None, fast_weights=None):
        if fast_weights:
            self.transform.weight = fast_weights[0]
            self.transform.bias = fast_weights[1]

        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), edge_weight)
        x = self.transform(x)
        x = F.relu(x)

        hidden = x
        for hop in range(self.num_layers):
            x = self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_feature)
            hidden = hidden + x
        return hidden

    def message(self, x_i, x_j, norm, edge_feature):
        return x_j * norm.view(-1, 1)

class WinGNN(nn.Module):
    def __init__(self, dim_in, hidden_dim, num_layer):
        super(WinGNN, self).__init__()
        self.num_layer = num_layer

        self.layer = WinGNNLayer(dim_in, hidden_dim, self.num_layer)

        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x, edge_index, edge_label_index, fast_weights=None):
        if fast_weights:
            self.linear1.weight = fast_weights[-4]
            self.linear1.bias = fast_weights[-3]
            self.linear2.weight = fast_weights[-2]
            self.linear2.bias = fast_weights[-1]

        x = self.layer(x, edge_index, fast_weights=fast_weights[0:2])
        x = F.normalize(x)

        x_i = x[edge_label_index[0]]
        x_j = x[edge_label_index[1]]

        prediction = torch.cat([x_i, x_j], dim=-1)
        prediction = self.linear1(prediction)
        prediction = F.relu(prediction)
        prediction = self.linear2(prediction)
        return prediction