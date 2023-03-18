import os
import sys
import time
import random
import numpy as np

from scipy.stats import ortho_group

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.datasets import QM9
from torch_scatter import scatter

from mpnn import MPNNLayer


class RMPNN(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, graph_out_dim=1, node_out_dim=5):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.graph_lin_pred = Linear(emb_dim, graph_out_dim)
        self.node_lin_pred = Linear(emb_dim, node_out_dim)

        self.hist_lin = Linear(emb_dim, emb_dim)
        
    def forward(self, data, prev_h):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)

        h = h + self.hist_lin(prev_h) # encode history
        h = torch.relu(h)
        
        #data.edge_attr = data.edge_attr.unsqueeze(1)
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = torch.mean(h, 0)
        #self.pool(h, torch.ones((1), dtype=torch.int64)) #data.batch) # (n, d) -> (batch_size, d)

        graph_out = self.graph_lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
        node_out = self.node_lin_pred(h)

        return graph_out, node_out, h
    
class RMPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, graph_out_dim=2, node_out_dim=2):
        super().__init__()
        self.sub_model = RMPNN(num_layers, emb_dim, in_dim, edge_dim, graph_out_dim, node_out_dim)
        self.emb_dim = emb_dim

    def forward(self, sequence):
        hidden = torch.zeros((self.emb_dim))
        graph_outs = []
        node_outs = []
        for t in range(len(sequence)):
            graph_out, node_out, hidden = self.sub_model(sequence[t],hidden)
            graph_outs.append(graph_out)
            node_outs.append(node_out)
        
        return graph_outs, node_outs
    

