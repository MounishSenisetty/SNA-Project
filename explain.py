"""
Explainability module for interpreting GNN predictions.
Implements GNNExplainer and Gradient Saliency methods.
"""

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer


class GradientSaliency:
    """
    Compute gradient-based saliency scores for node importance.
    """
    
    @staticmethod
    def compute_saliency(
        model: torch.nn.Module,
        data: Data,
        target_node: int,
        device: torch.device
    ) -> np.ndarray:
        """
        Compute gradient saliency for a target node.
        
        Args:
            model: Trained GNN model
            data: PyTorch Geometric data object
            target_node: Node index to explain
            device: Device to use
            
        Returns:
            Saliency scores for neighboring nodes
        """
        model.eval()
        data = data.to(device)
        
        # Enable gradient computation
        data.x.requires_grad_(True)
        
        # Forward pass
        out = model(data.x, data.edge_index)
        target_output = out[target_node]
        
        # Backward pass
        target_output.backward()
        
        # Get gradients
        saliency = torch.abs(data.x.grad).sum(dim=1).cpu().detach().numpy()
        
        # Normalize
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency
    
    @staticmethod
    def compute_edge_saliency(
        model: torch.nn.Module,
        data: Data,
        target_node: int,
        device: torch.device,
        top_k: int = 10
    ) -> List[Tuple[int, int, float]]:
        """
        Compute saliency scores for edges connected to target node.
        
        Args:
            model: Trained GNN model
            data: PyTorch Geometric data object
            target_node: Node index to explain
            device: Device to use
            top_k: Number of top edges to return
            
        Returns:
            List of (source, target, saliency) tuples
        """
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        
        edge_index = data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        
        # Get neighbors
        neighbors = list(G.neighbors(target_node))
        
        # Compute saliency for each neighbor
        edge_saliencies = []
        
        saliency = GradientSaliency.compute_saliency(model, data, target_node, device)
        
        for neighbor in neighbors:
            score = saliency[neighbor]
            edge_saliencies.append((target_node, neighbor, float(score)))
        
        # Sort by saliency score
        edge_saliencies.sort(key=lambda x: x[2], reverse=True)
        
        return edge_saliencies[:top_k]


class ExplainabilityAnalyzer:
    """
    Comprehensive explainability analysis for influencer predictions.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize explainability analyzer.
        
        Args:
            model: Trained GNN model
            device: Device to use
        """
        self.model = model
        self.device = device
    
    def get_important_neighbors(
        self,
        data: Data,
        target_node: int,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top-k important neighbors for a node.
        
        Args:
            data: PyTorch Geometric data object
            target_node: Node index
            k: Number of neighbors
            
        Returns:
            List of (neighbor_id, importance_score) tuples
        """
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        
        edge_index = data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        
        # Get neighbors
        neighbors = list(G.neighbors(target_node))
        
        # Compute feature importance
        saliency = GradientSaliency.compute_saliency(
            self.model,
            data,
            target_node,
            self.device
        )
        
        # Get importance scores for neighbors
        neighbor_scores = [(n, float(saliency[n])) for n in neighbors]
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        
        return neighbor_scores[:k]
    
    def get_important_edges(
        self,
        data: Data,
        target_node: int,
        k: int = 5
    ) -> List[Tuple[int, int, float]]:
        """
        Get top-k important edges for a node.
        
        Args:
            data: PyTorch Geometric data object
            target_node: Node index
            k: Number of edges
            
        Returns:
            List of (source, target, score) tuples
        """
        return GradientSaliency.compute_edge_saliency(
            self.model,
            data,
            target_node,
            self.device,
            top_k=k
        )
    
    def explain_prediction(
        self,
        data: Data,
        target_node: int,
        k_neighbors: int = 5
    ) -> Dict:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            data: PyTorch Geometric data object
            target_node: Node index to explain
            k_neighbors: Number of important neighbors
            
        Returns:
            Dictionary with explanation components
        """
        # Get model's prediction
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            prediction = float(out[target_node].item())
        
        # Get important neighbors
        important_neighbors = self.get_important_neighbors(data, target_node, k=k_neighbors)
        
        # Get important edges
        important_edges = self.get_important_edges(data, target_node, k=k_neighbors)
        
        # Get node features
        node_features = data.x[target_node].detach().cpu().numpy()
        
        explanation = {
            'target_node': target_node,
            'prediction': prediction,
            'important_neighbors': important_neighbors,
            'important_edges': important_edges,
            'node_features': node_features,
            'num_neighbors': len(list(nx.Graph(list(zip(data.edge_index[0].numpy(),
                                                        data.edge_index[1].numpy()))).neighbors(target_node)))
        }
        
        return explanation
    
    def explain_top_k_nodes(
        self,
        data: Data,
        y_pred: np.ndarray,
        k: int = 5,
        k_neighbors: int = 3
    ) -> List[Dict]:
        """
        Generate explanations for top-k predicted influential nodes.
        
        Args:
            data: PyTorch Geometric data object
            y_pred: Model predictions
            k: Number of top nodes to explain
            k_neighbors: Number of important neighbors per node
            
        Returns:
            List of explanation dictionaries
        """
        # Get top-k nodes
        top_k_indices = np.argsort(-y_pred)[:k]
        
        explanations = []
        for node_idx in top_k_indices:
            explanation = self.explain_prediction(data, int(node_idx), k_neighbors=k_neighbors)
            explanations.append(explanation)
        
        return explanations


def create_explanation_subgraph(
    data: Data,
    target_node: int,
    important_nodes: List[int]
) -> Data:
    """
    Create a subgraph containing target node and important neighbors.
    
    Args:
        data: Original PyTorch Geometric data
        target_node: Central node
        important_nodes: List of important neighboring nodes
        
    Returns:
        Subgraph data object
    """
    # Include target node and important nodes
    all_nodes = [target_node] + important_nodes
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
    
    # Filter edges
    edge_index = data.edge_index.numpy()
    mask = np.zeros(edge_index.shape[1], dtype=bool)
    
    for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
        if src in node_mapping and dst in node_mapping:
            mask[i] = True
    
    # Create subgraph
    new_edge_index = edge_index[:, mask]
    
    # Remap edges
    new_edge_index[0] = np.array([node_mapping[src] for src in new_edge_index[0]])
    new_edge_index[1] = np.array([node_mapping[dst] for dst in new_edge_index[1]])
    
    # Create new data object
    subgraph_data = Data(
        x=data.x[all_nodes],
        edge_index=torch.LongTensor(new_edge_index),
        y=data.y[all_nodes] if hasattr(data, 'y') else None
    )
    
    return subgraph_data, node_mapping


def get_node_importance_map(
    model: torch.nn.Module,
    data: Data,
    device: torch.device
) -> np.ndarray:
    """
    Get node-level importance map for the entire graph.
    
    Args:
        model: Trained model
        data: PyTorch Geometric data
        device: Device to use
        
    Returns:
        Importance scores for all nodes
    """
    model.eval()
    
    importance_map = np.zeros(data.num_nodes)
    
    with torch.no_grad():
        for node_idx in range(data.num_nodes):
            saliency = GradientSaliency.compute_saliency(
                model,
                data,
                node_idx,
                device
            )
            importance_map[node_idx] = saliency.mean()
    
    return importance_map
