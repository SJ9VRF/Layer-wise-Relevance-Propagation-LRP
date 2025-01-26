"""
Layer implementations for Layer-wise Relevance Propagation.

This module contains implementations of different neural network layers
that support Layer-wise Relevance Propagation for model interpretability.
"""

from typing import Optional, Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class LRPLayer(ABC):
    """Abstract base class for LRP-compatible layers.
    
    Attributes:
        prev_layer: Reference to previous layer in network
        next_layer: Reference to next layer in network
        activations: Stored activations from forward pass
        relevance: Stored relevance scores from backward pass
        name: Layer name for debugging
    """
    
    def __init__(self, name: str = ""):
        self.prev_layer: Optional[LRPLayer] = None
        self.next_layer: Optional[LRPLayer] = None
        self.activations: Optional[np.ndarray] = None
        self.relevance: Optional[np.ndarray] = None
        self.name = name

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass computation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output activations
        """
        pass

    @abstractmethod
    def backward(self, R: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Backward pass computing relevance.
        
        Args:
            R: Relevance from next layer
            eps: Small epsilon for numerical stability
            
        Returns:
            Relevance scores for current layer
        """
        pass
    
    def clean(self) -> None:
        """Reset stored activations and relevance."""
        self.activations = None 
        self.relevance = None

class LRPDense(LRPLayer):
    """LRP implementation for fully-connected layers.
    
    Implements LRP-ε rule: R_i = x_i * Σ_j (w_ij * R_j / (z_j + ε * sign(z_j)))
    
    Attributes:
        weights: Weight matrix
        biases: Bias vector
        z: Stored pre-activations from forward pass
        alpha: Alpha parameter for alpha-beta rule
        beta: Beta parameter for alpha-beta rule
        rule: LRP rule to use ('epsilon' or 'alpha_beta')
    """
    
    def __init__(self, 
                 weights: Union[np.ndarray, torch.Tensor],
                 biases: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 rule: str = 'epsilon',
                 alpha: float = 2.0,
                 beta: float = 1.0,
                 name: str = ""):
        """Initialize dense layer.
        
        Args:
            weights: Weight matrix (input_dim x output_dim)
            biases: Optional bias vector (output_dim)
            rule: LRP rule to use ('epsilon' or 'alpha_beta')
            alpha: Alpha parameter for alpha-beta rule
            beta: Beta parameter for alpha-beta rule
            name: Layer name for debugging
        """
        super().__init__(name)
        
        # Convert torch tensors to numpy if needed
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        if isinstance(biases, torch.Tensor):
            biases = biases.detach().cpu().numpy()
            
        self.weights = weights
        self.biases = biases if biases is not None else np.zeros(weights.shape[1])
        self.z: Optional[np.ndarray] = None
        
        # LRP rule parameters
        self.rule = rule
        self.alpha = alpha
        self.beta = beta
        
        # Validate parameters
        if rule not in ['epsilon', 'alpha_beta']:
            raise ValueError(f"Unknown LRP rule: {rule}")
        if rule == 'alpha_beta' and (alpha - beta) != 1:
            raise ValueError("Alpha-Beta parameters must satisfy alpha-beta=1")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass computing z = wx + b.
        
        Args:
            x: Input tensor (batch_size x input_dim)
            
        Returns:
            Output tensor (batch_size x output_dim)
        """
        self.activations = x
        z = np.dot(x, self.weights) + self.biases
        self.z = z
        return z

    def backward(self, R: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Backward pass computing relevance scores.
        
        Args:
            R: Relevance scores from next layer
            eps: Small epsilon for numerical stability
            
        Returns:
            Relevance scores for input neurons
        """
        self.relevance = R
        
        if self.rule == 'epsilon':
            return self._backward_epsilon(R, eps)
        else:
            return self._backward_alphabeta(R, eps)

    def _backward_epsilon(self, R: np.ndarray, eps: float) -> np.ndarray:
        """LRP-ε rule implementation."""
        z = self.z
        denominator = z + eps * ((z >= 0).astype(float) * 2 - 1)
        return self.activations * np.dot((R / denominator), self.weights.T)

    def _backward_alphabeta(self, R: np.ndarray, eps: float) -> np.ndarray:
        """LRP-αβ rule implementation."""
        z = self.z
        weights_pos = np.maximum(0, self.weights)
        weights_neg = np.minimum(0, self.weights)
        
        # Positive relevance
        z_pos = np.dot(self.activations, weights_pos) 
        denom_pos = z_pos + eps
        rel_pos = self.activations * np.dot((R * self.alpha / denom_pos), weights_pos.T)
        
        # Negative relevance
        z_neg = np.dot(self.activations, weights_neg)
        denom_neg = z_neg - eps
        rel_neg = self.activations * np.dot((R * self.beta / denom_neg), weights_neg.T)
        
        return rel_pos - rel_neg

class LRPReLU(LRPLayer):
    """LRP implementation for ReLU activation.
    
    Simply propagates relevance through active neurons.
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass applying ReLU.
        
        Args:
            x: Input tensor
            
        Returns:
            ReLU(x)
        """
        self.activations = x
        return np.maximum(0, x)

    def backward(self, R: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Backward pass propagating relevance through active neurons.
        
        Args:
            R: Relevance from next layer
            eps: Unused, kept for API consistency
            
        Returns:
            Relevance scores masked by ReLU activation
        """
        self.relevance = R
        return R * (self.activations > 0)

class LRPDropout(LRPLayer):
    """LRP implementation for dropout layer.
    
    Stores and applies dropout mask during training.
    """
    
    def __init__(self, p: float = 0.5, name: str = ""):
        """Initialize dropout layer.
        
        Args:
            p: Dropout probability
            name: Layer name for debugging
        """
        super().__init__(name)
        self.p = p
        self.mask: Optional[np.ndarray] = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass applying dropout mask.
        
        Args:
            x: Input tensor
            
        Returns:
            Masked tensor during training, original tensor during inference
        """
        self.activations = x
        
        if self.training:
            self.mask = (np.random.random(x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, R: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Backward pass applying stored mask.
        
        Args:
            R: Relevance from next layer
            eps: Unused, kept for API consistency
            
        Returns:
            Masked relevance scores during training, original scores during inference
        """
        self.relevance = R
        if self.training and self.mask is not None:
            return R * self.mask
        return R

class LRPBatchNorm(LRPLayer):
    """LRP implementation for batch normalization.
    
    Applies learned scale and shift while tracking running statistics.
    """
    
    def __init__(self, 
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 name: str = ""):
        """Initialize batch norm layer.
        
        Args:
            num_features: Number of input features
            eps: Small epsilon for numerical stability
            momentum: Momentum for running statistics
            affine: Whether to apply learnable affine transform
            name: Layer name for debugging
        """
        super().__init__(name)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Learnable parameters
        if affine:
            self.weight = np.ones(num_features)
            self.bias = np.zeros(num_features)
        
        # Storage for forward pass
        self.x_normalized: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass applying batch normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        self.activations = x
        
        if self.training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        self.std = np.sqrt(var + self.eps)
        self.x_normalized = (x - mean) / self.std
        
        if self.affine:
            return self.weight * self.x_normalized + self.bias
        return self.x_normalized

    def backward(self, R: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Backward pass computing relevance through normalization.
        
        Args:
            R: Relevance from next layer
            eps: Small epsilon for numerical stability
            
        Returns:
            Relevance scores for input features
        """
        self.relevance = R
        
        if self.affine:
            # Remove affine transform
            R = R * self.weight
            
        if self.x_normalized is not None and self.std is not None:
            # Redistribute through standardization
            return R / (self.std + eps)
        return R

def convert_torch_layer(module: nn.Module) -> LRPLayer:
    """Convert PyTorch module to LRP layer.
    
    Args:
        module: PyTorch module to convert
        
    Returns:
        Equivalent LRP layer
        
    Raises:
        ValueError: If module type is not supported
    """
    if isinstance(module, nn.Linear):
        return LRPDense(
            weights=module.weight.data,
            biases=module.bias.data if module.bias is not None else None,
            name=module.__class__.__name__
        )
    elif isinstance(module, nn.ReLU):
        return LRPReLU(name=module.__class__.__name__)
    elif isinstance(module, nn.Dropout):
        return LRPDropout(p=module.p, name=module.__class__.__name__)
    elif isinstance(module, nn.BatchNorm1d):
        layer = LRPBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            name=module.__class__.__name__
        )
        if module.affine:
            layer.weight = module.weight.data.numpy()
            layer.bias = module.bias.data.numpy()
        layer.running_mean = module.running_mean.data.numpy()
        layer.running_var = module.running_var.data.numpy()
        return layer
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
