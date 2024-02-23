import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

from model.layers import LinkDecoder

class TGCN(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, num_layer):
        super(TGCN, self).__init__()

        self.num_layer = num_layer
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out

        self.node_embedding = nn.Sequential(nn.Linear(self.dim_in, self.hidden_dim),
                                           nn.ReLU())

        self.layer = nn.ModuleList([TGCNLayer(self.hidden_dim)
                                    for i in range(self.num_layer)])

        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)

    def forward(self, x, edge_index, edge_label_index, H):
        x = self.node_embedding(x)

        for i, layer in enumerate(self.layer):
            x = layer(x, edge_index, H=H)

        prediction = self.decoder(x, edge_label_index)
        return prediction, x
    
class TGCNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super(TGCNLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.layer = T_GCN(self.hidden_dim, self.hidden_dim)

        self.post_layer = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(self.dropout))

    def forward(self, x, edge_index, edge_weight=None, H=None):
        x = self.layer(x, edge_index, edge_weight, H=H)
        x = self.post_layer(x)
        return x
    
class T_GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(T_GCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H