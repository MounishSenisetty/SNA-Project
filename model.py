import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GATConv, GCNConv, SAGEConv


class GCNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        assert num_layers >= 2, "Use at least 2 layers (encoder + output)."

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)  # [N, 1]
        return x


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(num_layers - 1, 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        del edge_index
        return self.net(x)


class SAGERegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        assert num_layers >= 2
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GATRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2, heads: int = 2, out_dim: int = 1):
        super().__init__()
        assert num_layers >= 2
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.out = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)


class APPNPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2, k: int = 10, alpha: float = 0.1, out_dim: int = 1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.prop = APPNP(K=k, alpha=alpha)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x


def build_model(name: str, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int = 1):
    name = name.lower()
    if name == "gcn":
        return GCNRegressor(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, out_dim=out_dim)
    if name == "mlp":
        return MLPRegressor(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, out_dim=out_dim)
    if name == "graphsage":
        return SAGERegressor(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, out_dim=out_dim)
    if name == "gat":
        return GATRegressor(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, out_dim=out_dim)
    if name == "appnp":
        return APPNPRegressor(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, out_dim=out_dim)
    raise ValueError(f"Unsupported model: {name}")