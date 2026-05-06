"""
Graph Neural Network models for influencer prediction.
Implements GCN (mandatory), GraphSAGE, and GAT (optional).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from typing import Optional, Dict


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) for node-level prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for regression)
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Node predictions
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation for regression)
        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for node-level prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for regression)
            num_layers: Number of SAGE layers
            dropout: Dropout rate
        """
        super(GraphSAGEModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Node predictions
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation for regression)
        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) model for node-level prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_heads: int = 4
    ):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for regression)
            num_layers: Number of GAT layers
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super(GATModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_dim * num_heads,
                hidden_dim,
                heads=num_heads,
                dropout=dropout
            ))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False))
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Node predictions
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation for regression)
        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)


class ModelFactory:
    """
    Factory for creating different GNN models.
    """
    
    MODELS = {
        'gcn': GCNModel,
        'graphsage': GraphSAGEModel,
        'gat': GATModel
    }
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        **kwargs
    ) -> nn.Module:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model ('gcn', 'graphsage', 'gat')
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        if model_name.lower() not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODELS.keys())}")
        
        model_class = cls.MODELS[model_name.lower()]
        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    
    @classmethod
    def list_models(cls) -> list:
        """List available models."""
        return list(cls.MODELS.keys())
