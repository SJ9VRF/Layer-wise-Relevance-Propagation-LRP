"""
Network implementation for Layer-wise Relevance Propagation.

This module provides the core network functionality for LRP, including
model construction, forward/backward passes, and PyTorch conversion utilities.
"""

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings

from .layers import LRPLayer, convert_torch_layer

class LRPNetwork:
    """Neural network with Layer-wise Relevance Propagation capability.
    
    Attributes:
        layers: Ordered dictionary of LRP-compatible layers
        input_shape: Expected input shape
        output_shape: Network output shape
        training: Whether network is in training mode
    """
    
    def __init__(self, 
                 layers: Union[List[LRPLayer], Dict[str, LRPLayer]],
                 input_shape: Optional[Tuple[int, ...]] = None,
                 name: str = ""):
        """Initialize LRP network.
        
        Args:
            layers: List or dict of LRP layers
            input_shape: Optional input shape for validation
            name: Network name for debugging
        """
        # Convert list to ordered dict if needed
        if isinstance(layers, list):
            self.layers = OrderedDict([
                (f"layer_{i}", layer) for i, layer in enumerate(layers)
            ])
        else:
            self.layers = OrderedDict(layers)
            
        self.input_shape = input_shape
        self.name = name
        self.training = True
        
        # Link layers
        prev_layer = None
        for layer in self.layers.values():
            if prev_layer is not None:
                layer.prev_layer = prev_layer
                prev_layer.next_layer = layer
            prev_layer = layer
            
        # Determine output shape if possible
        self.output_shape = self._infer_output_shape()

    def forward(self, x: np.ndarray, training: Optional[bool] = None) -> np.ndarray:
        """Forward pass through network.
        
        Args:
            x: Input tensor
            training: Optional override for training mode
            
        Returns:
            Network output
            
        Raises:
            ValueError: If input shape doesn't match expected shape
        """
        if training is not None:
            prev_mode = self.training
            self.training = training
            
        try:
            # Input shape validation
            if self.input_shape is not None:
                if x.shape[1:] != self.input_shape:
                    raise ValueError(
                        f"Input shape {x.shape[1:]} doesn't match "
                        f"expected shape {self.input_shape}"
                    )
            
            # Forward pass
            current = x
            for layer in self.layers.values():
                if hasattr(layer, 'training'):
                    layer.training = self.training
                current = layer.forward(current)
                
            return current
            
        finally:
            if training is not None:
                self.training = prev_mode

    def explain(self, 
                x: np.ndarray,
                target_class: Optional[Union[int, np.ndarray]] = None,
                output_relevance: Optional[np.ndarray] = None,
                eps: float = 1e-9,
                layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Generate LRP explanation.
        
        Args:
            x: Input to explain
            target_class: Target class index or one-hot vector
            output_relevance: Optional custom relevance for output layer
            eps: Small epsilon for numerical stability
            layer_names: Optional list of layer names to get relevance for
            
        Returns:
            Dictionary mapping layer names to relevance scores
            
        Raises:
            ValueError: If neither target_class nor output_relevance is provided
        """
        # Forward pass
        prediction = self.forward(x, training=False)
        
        # Initialize output relevance
        if output_relevance is None:
            if target_class is None:
                raise ValueError("Must provide either target_class or output_relevance")
                
            R = np.zeros_like(prediction)
            if isinstance(target_class, np.ndarray):
                R = prediction * target_class
            else:
                R[..., target_class] = prediction[..., target_class]
        else:
            R = output_relevance
            
        # Backward pass collecting relevance
        relevance_maps = {}
        for name, layer in reversed(list(self.layers.items())):
            if layer_names is None or name in layer_names:
                relevance_maps[name] = R.copy()
            R = layer.backward(R, eps)
            
        # Add input relevance
        relevance_maps['input'] = R
        
        return relevance_maps

    def train(self) -> 'LRPNetwork':
        """Set network to training mode."""
        self.training = True
        return self

    def eval(self) -> 'LRPNetwork':
        """Set network to evaluation mode."""
        self.training = False
        return self

    def get_layer(self, name: str) -> LRPLayer:
        """Get layer by name.
        
        Args:
            name: Layer name
            
        Returns:
            Requested layer
            
        Raises:
            KeyError: If layer doesn't exist
        """
        return self.layers[name]

    def add_layer(self, name: str, layer: LRPLayer) -> None:
        """Add new layer to network.
        
        Args:
            name: Layer name
            layer: Layer to add
            
        Raises:
            ValueError: If layer name already exists
        """
        if name in self.layers:
            raise ValueError(f"Layer {name} already exists")
            
        # Link with previous layer if it exists
        if len(self.layers) > 0:
            prev_layer = list(self.layers.values())[-1]
            layer.prev_layer = prev_layer
            prev_layer.next_layer = layer
            
        self.layers[name] = layer
        
        # Update output shape
        self.output_shape = self._infer_output_shape()

    def remove_layer(self, name: str) -> None:
        """Remove layer from network.
        
        Args:
            name: Name of layer to remove
            
        Raises:
            KeyError: If layer doesn't exist
        """
        layer = self.layers[name]
        
        # Update layer links
        if layer.prev_layer is not None:
            layer.prev_layer.next_layer = layer.next_layer
        if layer.next_layer is not None:
            layer.next_layer.prev_layer = layer.prev_layer
            
        del self.layers[name]
        
        # Update output shape
        self.output_shape = self._infer_output_shape()

    def clean(self) -> None:
        """Reset stored activations and relevance in all layers."""
        for layer in self.layers.values():
            layer.clean()

    def _infer_output_shape(self) -> Optional[Tuple[int, ...]]:
        """Infer output shape if possible."""
        if self.input_shape is None:
            return None
            
        try:
            x = np.zeros((1,) + self.input_shape)
            out = self.forward(x, training=False)
            return tuple(out.shape[1:])
        except:
            return None

    def save_state(self) -> Dict[str, Any]:
        """Save network state including weights and running statistics.
        
        Returns:
            Dictionary containing network state
        """
        state = {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'name': self.name,
            'layers': {}
        }
        
        for name, layer in self.layers.items():
            layer_state = {}
            
            # Save common attributes
            for attr in ['weights', 'biases', 'running_mean', 'running_var',
                        'weight', 'bias', 'p', 'eps', 'momentum']:
                if hasattr(layer, attr):
                    layer_state[attr] = getattr(layer, attr)
                    
            state['layers'][name] = layer_state
            
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load network state.
        
        Args:
            state: State dictionary from save_state()
            
        Raises:
            ValueError: If state doesn't match network architecture
        """
        if set(state['layers'].keys()) != set(self.layers.keys()):
            raise ValueError("State layers don't match network architecture")
            
        self.input_shape = state['input_shape']
        self.output_shape = state['output_shape']
        self.name = state['name']
        
        for name, layer_state in state['layers'].items():
            layer = self.layers[name]
            for attr, value in layer_state.items():
                setattr(layer, attr, value)

def convert_torch_to_lrp(model: nn.Module,
                        input_shape: Optional[Tuple[int, ...]] = None,
                        custom_layer_map: Optional[Dict[str, LRPLayer]] = None) -> LRPNetwork:
    """Convert PyTorch model to LRP network.
    
    Args:
        model: PyTorch model to convert
        input_shape: Optional input shape for validation
        custom_layer_map: Optional mapping of layer names to custom LRP layers
        
    Returns:
        Equivalent LRP network
        
    Raises:
        ValueError: If unsupported layer type is encountered
    """
    layers = OrderedDict()
    custom_layer_map = custom_layer_map or {}
    
    def _add_module(module: nn.Module, name: str) -> None:
        """Recursively add modules to layers dict."""
        if name in custom_layer_map:
            layers[name] = custom_layer_map[name]
        elif len(list(module.children())) == 0:  # Leaf module
            try:
                layers[name] = convert_torch_layer(module)
            except ValueError as e:
                warnings.warn(f"Skipping layer {name}: {str(e)}")
        else:  # Container module
            for child_name, child in module.named_children():
                _add_module(child, f"{name}.{child_name}" if name else child_name)
                
    _add_module(model, "")
    
    return LRPNetwork(
        layers=layers,
        input_shape=input_shape,
        name=model.__class__.__name__
    )

class SequentialLRP(LRPNetwork):
    """Convenience class for sequential LRP networks."""
    
    def __init__(self, *layers: LRPLayer, input_shape: Optional[Tuple[int, ...]] = None):
        """Initialize sequential network.
        
        Args:
            *layers: Variable number of LRP layers
            input_shape: Optional input shape for validation
        """
        super().__init__(list(layers), input_shape)

def create_classifier(input_dim: int,
                     hidden_dims: List[int],
                     num_classes: int,
                     dropout: float = 0.0) -> LRPNetwork:
    """Create simple classifier network with LRP support.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        dropout: Dropout probability
        
    Returns:
        LRP-compatible classifier network
    """
    from .layers import LRPDense, LRPReLU, LRPDropout
    
    layers = []
    prev_dim = input_dim
    
    for i, dim in enumerate(hidden_dims):
        layers.extend([
            (f"dense_{i}", LRPDense(
                weights=np.random.randn(prev_dim, dim) / np.sqrt(prev_dim),
                biases=np.zeros(dim)
            )),
            (f"relu_{i}", LRPReLU())
        ])
        if dropout > 0:
            layers.append((f"dropout_{i}", LRPDropout(p=dropout)))
        prev_dim = dim
        
    layers.append((
        "output",
        LRPDense(
            weights=np.random.randn(prev_dim, num_classes) / np.sqrt(prev_dim),
            biases=np.zeros(num_classes)
        )
    ))
    
    return LRPNetwork(
        layers=OrderedDict(layers),
        input_shape=(input_dim,)
    )
