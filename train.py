"""
Training module for GNN models.
Includes early stopping, learning rate scheduling, and checkpoint management.
"""

import torch
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Callable
import json


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelTrainer:
    """
    Trainer for GNN models with early stopping and checkpoints.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to use (CPU or GPU)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, data: Data, train_mask: torch.Tensor) -> float:
        """
        Train for one epoch.
        
        Args:
            data: PyTorch Geometric data object
            train_mask: Boolean mask for training nodes
            
        Returns:
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return float(loss.item())
    
    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Tuple[float, Dict]:
        """
        Evaluate model on a dataset.
        
        Args:
            data: PyTorch Geometric data object
            mask: Boolean mask for nodes to evaluate
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        
        out = self.model(data.x, data.edge_index)
        loss = torch.nn.functional.mse_loss(out[mask], data.y[mask])
        
        # Calculate additional metrics
        y_true = data.y[mask].cpu().numpy()
        y_pred = out[mask].cpu().numpy()
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse)
        }
        
        return float(loss.item()), metrics
    
    def train(
        self,
        data: Data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            data: PyTorch Geometric data object
            train_mask: Training nodes mask
            val_mask: Validation nodes mask
            test_mask: Test nodes mask
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
            
        Returns:
            Dictionary with training history and final metrics
        """
        # Move data to device
        data = data.to(self.device)
        self.model = self.model.to(self.device)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        
        # Training loop
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(data, train_mask)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.evaluate(data, val_mask)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping
            if early_stopping(val_loss):
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # Update progress bar
            if verbose and (epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        test_loss, test_metrics = self.evaluate(data, test_mask)
        
        # Get final predictions
        self.model.eval()
        with torch.no_grad():
            final_out = self.model(data.x, data.edge_index)
        
        results = {
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss),
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'final_predictions': final_out.cpu().numpy(),
            'history': self.history
        }
        
        return results
    
    def save_checkpoint(self, filepath: str, metadata: Dict = None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            metadata: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


def train_model(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    early_stopping_patience: int = 20,
    checkpoint_dir: str = './checkpoints',
    verbose: bool = True
) -> Dict:
    """
    Convenience function to train a model.
    
    Args:
        model: PyTorch model
        data: PyTorch Geometric data object
        train_mask: Training nodes mask
        val_mask: Validation nodes mask
        test_mask: Test nodes mask
        device: Device to use
        learning_rate: Learning rate
        weight_decay: Weight decay
        epochs: Number of epochs
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory for checkpoints
        verbose: Print progress
        
    Returns:
        Training results
    """
    trainer = ModelTrainer(
        model,
        device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir
    )
    
    results = trainer.train(
        data,
        train_mask,
        val_mask,
        test_mask,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose
    )
    
    return results
