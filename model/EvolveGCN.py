from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import GRU

from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from model.layers import LinkDecoder

class EvolveGCN(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, num_layer, num_nodes, rnn='LSTM'):
        super(EvolveGCN, self).__init__()

        self.num_layer = num_layer
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.rnn = rnn

        self.node_embedding = nn.Sequential(nn.Linear(self.dim_in, self.hidden_dim),
                                            nn.ReLU())

        self.layer = nn.ModuleList([EGCNLayer(self.hidden_dim, num_nodes, rnn=self.rnn)
                                    for i in range(self.num_layer)])

        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)

    def forward(self, x, edge_index, edge_label_index):
        x = self.node_embedding(x)

        for i, layer in enumerate(self.layer):
            x = layer(x, edge_index)

        prediction = self.decoder(x, edge_label_index)
        return prediction
    
class EGCNLayer(nn.Module):
    def __init__(self, hidden_dim, num_nodes, dropout=0.0, rnn='LSTM'):
        super(EGCNLayer, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if rnn == 'LSTM':
            self.layer = EvolveGCNO(self.hidden_dim)
        else:
            raise NotImplementedError('No such RNN model {}'.format(rnn))

        self.post_layer = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(self.dropout))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.layer(x, edge_index, edge_weight)
        x = self.post_layer(x)
        return x
    
class GCNConv_Fixed_W(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv_Fixed_W, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, W: torch.FloatTensor, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = torch.matmul(x, W)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class EvolveGCNO(torch.nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(EvolveGCNO, self).__init__()

        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.initial_weight = torch.nn.Parameter(torch.Tensor(1, in_channels, in_channels))
        self.weight = None
        self._create_layers()
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.initial_weight)

    def reinitialize_weight(self):
        self.weight = None

    def _create_layers(self):

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )
        for param in self.recurrent_layer.parameters():
            param.requires_grad = True
            param.retain_grad()

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        
        if self.weight is None:
            _, self.weight = self.recurrent_layer(self.initial_weight, self.initial_weight)
        else:
            _, self.weight = self.recurrent_layer(self.weight, self.weight)
        self.weight = self.weight.detach()
        X = self.conv_layer(self.weight.squeeze(dim=0), X, edge_index, edge_weight)
        return X