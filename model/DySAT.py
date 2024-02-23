import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATConv
from model.layers import LinkDecoder

class TemporalAttentionLayer(nn.Module):
    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual):
        super(TemporalAttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))

        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        self.attn_dp = nn.Dropout(attn_drop)

        self.xavier_init()

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

    def forward(self, inputs):
        temporal_inputs = inputs

        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0]))
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0]))
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0]))

        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)
        
        outputs = torch.matmul(q_, k_.permute(0, 2, 1))
        outputs = outputs / (self.num_time_steps ** 0.5)

        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)
        padding = torch.ones_like(masks) * (-2**32 + 1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs
                
        if self.training:
            outputs = self.attn_dp(outputs)

        outputs = torch.matmul(outputs, v_)
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2)
        
        outputs = self.feedforward(outputs)

        if self.residual:
            outputs = outputs + temporal_inputs

        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

class DySAT(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, num_layer, time_step):
        super(DySAT, self).__init__()

        self.num_layer = num_layer
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out

        self.node_embedding = nn.Sequential(nn.Linear(self.dim_in, self.hidden_dim),
                                           nn.ReLU())
        
        self.spatial_layer = GATConv(self.hidden_dim, self.hidden_dim//2, heads=2)

        self.temporal_layer = TemporalAttentionLayer(self.hidden_dim, n_heads=2, num_time_steps=time_step, attn_drop=0.0, residual=True)

        self.decoder = LinkDecoder(self.hidden_dim, self.dim_out)
    
    def forward(self, x, edge_index, edge_label_index, history):
        x = self.node_embedding(x)
        x = self.spatial_layer(x, edge_index)
        history.append(x.unsqueeze(dim=1))
        temporal_input = torch.cat(history, dim=1)
        x = self.temporal_layer(temporal_input)
        prediction = self.decoder(x, edge_label_index)
        return prediction, temporal_input