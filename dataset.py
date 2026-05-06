"""
Dataset loading and centrality label generation module.
Supports Cora, CiteSeer, and Karate Club datasets.
Generates labels using Degree, PageRank, and Eigenvector centrality.
"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.data import Data
from typing import Dict, Tuple
import os
from utils import normalize_scores, ensure_dir


class InfluencerDataset:
    """
    Loads graph datasets and generates centrality-based influence labels.
    """
    
    def __init__(self, name: str = 'Cora', root: str = './data'):
        """
        Initialize the dataset loader.
        
        Args:
            name: Dataset name ('Cora', 'CiteSeer', 'Karate')
            root: Root directory for downloading datasets
        """
        self.name = name
        self.root = ensure_dir(root)
        self.data = None
        self.centrality_labels = {}
        self.G = None  # NetworkX graph representation
        
    def load_dataset(self) -> Data:
        """
        Load the specified dataset.
        
        Returns:
            PyTorch Geometric Data object
        """
        print(f"Loading {self.name} dataset...")
        
        if self.name in ['Cora', 'CiteSeer', 'PubMed']:
            # Use Planetoid datasets
            dataset = Planetoid(
                root=self.root,
                name=self.name,
                split='public'
            )
            self.data = dataset[0]
        elif self.name == 'Karate':
            # Use Karate Club dataset
            try:
                dataset = KarateClub(root=self.root)
            except TypeError:
                # Newer versions of PyG don't accept root parameter
                dataset = KarateClub()
            self.data = dataset[0]
        else:
            raise ValueError(f"Unknown dataset: {self.name}")
        
        print(f"Loaded {self.name}: {self.data.num_nodes} nodes, {self.data.num_edges} edges")
        return self.data
    
    def create_networkx_graph(self) -> nx.Graph:
        """
        Convert PyTorch Geometric data to NetworkX graph.
        
        Returns:
            NetworkX undirected graph
        """
        if self.data is None:
            self.load_dataset()
        
        G = nx.Graph()
        G.add_nodes_from(range(self.data.num_nodes))
        
        # Add edges from edge_index
        edge_index = self.data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        
        self.G = G
        return G
    
    def compute_centrality(self, method: str = 'degree') -> np.ndarray:
        """
        Compute node centrality using specified method.
        
        Args:
            method: 'degree', 'pagerank', or 'eigenvector'
            
        Returns:
            Normalized centrality scores (0-1)
        """
        if self.G is None:
            self.create_networkx_graph()
        
        print(f"Computing {method} centrality...")
        
        if method == 'degree':
            # Degree centrality
            centrality = nx.degree_centrality(self.G)
        elif method == 'pagerank':
            # PageRank centrality
            centrality = nx.pagerank(self.G)
        elif method == 'eigenvector':
            # Eigenvector centrality (handle disconnected graphs)
            try:
                centrality = nx.eigenvector_centrality_numpy(self.G, max_iter=1000)
            except:
                print(f"Warning: Eigenvector centrality failed, using degree centrality instead")
                centrality = nx.degree_centrality(self.G)
        else:
            raise ValueError(f"Unknown centrality method: {method}")
        
        # Convert to numpy array
        scores = np.array([centrality.get(i, 0.0) for i in range(self.data.num_nodes)])
        
        # Normalize to [0, 1]
        scores = normalize_scores(scores, min_val=0.0, max_val=1.0)
        
        self.centrality_labels[method] = scores
        return scores
    
    def generate_all_centralities(self) -> Dict[str, np.ndarray]:
        """
        Generate all three centrality measures.
        
        Returns:
            Dictionary mapping centrality method to scores
        """
        centralities = {}
        for method in ['degree', 'pagerank', 'eigenvector']:
            centralities[method] = self.compute_centrality(method)
        
        return centralities
    
    def get_labels(self, method: str = 'pagerank') -> torch.Tensor:
        """
        Get centrality labels as torch tensor.
        
        Args:
            method: Centrality method to use
            
        Returns:
            Torch tensor of normalized centrality scores
        """
        if method not in self.centrality_labels:
            self.compute_centrality(method)
        
        scores = self.centrality_labels[method]
        return torch.FloatTensor(scores)
    
    def get_data_with_labels(self, centrality_method: str = 'pagerank') -> Data:
        """
        Get PyTorch Geometric data with computed labels.
        
        Args:
            centrality_method: Method to use for label generation
            
        Returns:
            Data object with y attribute set to centrality scores
        """
        if self.data is None:
            self.load_dataset()
        
        # Generate labels if not already done
        if centrality_method not in self.centrality_labels:
            self.compute_centrality(centrality_method)
        
        # Create copy and add labels
        data = self.data.clone()
        data.y = self.get_labels(centrality_method)
        
        return data
    
    def info(self):
        """Print dataset information."""
        if self.data is None:
            self.load_dataset()
        
        print(f"\n{'='*50}")
        print(f"Dataset: {self.name}")
        print(f"Nodes: {self.data.num_nodes}")
        print(f"Edges: {self.data.num_edges}")
        print(f"Features: {self.data.num_features}")
        print(f"Average degree: {2 * self.data.num_edges / self.data.num_nodes:.2f}")
        
        if hasattr(self.data, 'x'):
            print(f"Feature shape: {self.data.x.shape}")
        
        print(f"{'='*50}\n")


def load_influencer_dataset(
    dataset_name: str = 'Cora',
    root: str = './data',
    centrality_method: str = 'pagerank'
) -> Tuple[Data, Dict[str, np.ndarray]]:
    """
    Convenient function to load dataset with centrality labels.
    
    Args:
        dataset_name: Name of the dataset
        root: Root directory for data
        centrality_method: Which centrality to use as labels
        
    Returns:
        Tuple of (PyTorch Geometric Data, centrality_labels dict)
    """
    dataset = InfluencerDataset(name=dataset_name, root=root)
    data = dataset.load_dataset()
    centralities = dataset.generate_all_centralities()
    
    # Add labels to data
    data.y = dataset.get_labels(centrality_method)
    
    dataset.info()
    
    return data, centralities
