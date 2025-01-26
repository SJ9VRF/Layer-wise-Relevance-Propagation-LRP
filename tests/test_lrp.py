"""
Comprehensive test suite for Layer-wise Relevance Propagation implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest
from typing import Tuple, List

from lrp.layers import (
    LRPLayer, LRPDense, LRPReLU, LRPDropout, LRPBatchNorm,
    convert_torch_layer
)
from lrp.network import (
    LRPNetwork, convert_torch_to_lrp, SequentialLRP, create_classifier
)

def generate_random_data(shape: Tuple[int, ...]) -> np.ndarray:
    """Generate random test data."""
    return np.random.randn(*shape)

class TestLRPLayers:
    """Test suite for individual LRP layer implementations."""
    
    def test_dense_forward(self):
        """Test dense layer forward pass."""
        weights = np.array([[1, 2], [3, 4]])
        biases = np.array([0.1, 0.2])
        layer = LRPDense(weights, biases)
        
        x = np.array([[1, 2]])
        expected = np.array([[7.1, 10.2]])
        
        np.testing.assert_array_almost_equal(layer.forward(x), expected)

    def test_dense_backward_epsilon(self):
        """Test dense layer backward pass with epsilon rule."""
        weights = np.array([[1, 2], [3, 4]])
        layer = LRPDense(weights, np.zeros(2))
        
        x = np.array([[1, 2]])
        layer.forward(x)
        R = np.array([[1, 2]])
        
        # Verify conservation property
        backward_R = layer.backward(R)
        np.testing.assert_array_almost_equal(
            np.sum(backward_R), np.sum(R)
        )

    def test_dense_backward_alphabeta(self):
        """Test dense layer backward pass with alpha-beta rule."""
        weights = np.array([[1, 2], [3, 4]])
        layer = LRPDense(weights, np.zeros(2), rule='alpha_beta')
        
        x = np.array([[1, 2]])
        layer.forward(x)
        R = np.array([[1, 2]])
        
        backward_R = layer.backward(R)
        np.testing.assert_array_almost_equal(
            np.sum(backward_R), np.sum(R)
        )

    def test_relu_forward(self):
        """Test ReLU layer forward pass."""
        layer = LRPReLU()
        x = np.array([[-1, 0, 1]])
        expected = np.array([[0, 0, 1]])
        
        np.testing.assert_array_equal(layer.forward(x), expected)

    def test_relu_backward(self):
        """Test ReLU layer backward pass."""
        layer = LRPReLU()
        x = np.array([[-1, 0, 1]])
        layer.forward(x)
        R = np.array([[1, 1, 1]])
        
        expected = np.array([[0, 0, 1]])
        np.testing.assert_array_equal(layer.backward(R), expected)

    def test_dropout(self):
        """Test dropout layer."""
        layer = LRPDropout(p=0.5)
        x = np.ones((100, 10))
        
        # Test training mode
        layer.training = True
        y_train = layer.forward(x)
        assert 0.4 < np.mean(y_train > 0) < 0.6
        
        # Test evaluation mode
        layer.training = False
        y_eval = layer.forward(x)
        np.testing.assert_array_equal(y_eval, x)

    def test_batchnorm(self):
        """Test batch normalization layer."""
        layer = LRPBatchNorm(num_features=2)
        x = np.array([[1, 2], [3, 4]])
        
        # Test forward pass
        y = layer.forward(x)
        assert y.shape == x.shape
        
        # Test backward pass
        R = np.ones_like(x)
        backward_R = layer.backward(R)
        np.testing.assert_array_almost_equal(
            np.sum(backward_R), np.sum(R)
        )

class TestLRPNetwork:
    """Test suite for LRP network functionality."""
    
    @pytest.fixture
    def simple_network(self) -> LRPNetwork:
        """Create simple test network."""
        layers = [
            LRPDense(np.eye(3), np.zeros(3)),
            LRPReLU(),
            LRPDense(np.ones((3, 1)), np.zeros(1))
        ]
        return LRPNetwork(layers, input_shape=(3,))

    def test_forward(self, simple_network):
        """Test network forward pass."""
        x = np.array([[1, -1, 2]])
        y = simple_network.forward(x)
        assert y.shape == (1, 1)

    def test_explain(self, simple_network):
        """Test network explanation generation."""
        x = np.array([[1, -1, 2]])
        relevance = simple_network.explain(x, target_class=0)
        
        assert 'input' in relevance
        assert len(relevance) == len(simple_network.layers) + 1
        
        # Test conservation property
        for R in relevance.values():
            np.testing.assert_array_almost_equal(
                np.sum(R), simple_network.forward(x)[0, 0]
            )

    def test_train_eval(self, simple_network):
        """Test training/evaluation mode switching."""
        assert simple_network.training
        
        simple_network.eval()
        assert not simple_network.training
        
        simple_network.train()
        assert simple_network.training

    def test_layer_manipulation(self, simple_network):
        """Test layer addition and removal."""
        # Add layer
        new_layer = LRPReLU()
        simple_network.add_layer("new_relu", new_layer)
        assert "new_relu" in simple_network.layers
        
        # Remove layer
        simple_network.remove_layer("new_relu")
        assert "new_relu" not in simple_network.layers

    def test_state_save_load(self, simple_network):
        """Test state saving and loading."""
        # Modify network state
        for layer in simple_network.layers.values():
            if isinstance(layer, LRPDense):
                layer.weights += 1
        
        # Save state
        state = simple_network.save_state()
        
        # Create new network and load state
        new_network = LRPNetwork(
            [LRPDense(np.eye(3), np.zeros(3)),
             LRPReLU(),
             LRPDense(np.ones((3, 1)), np.zeros(1))],
            input_shape=(3,)
        )
        new_network.load_state(state)
        
        # Test forward pass equivalence
        x = np.array([[1, -1, 2]])
        np.testing.assert_array_almost_equal(
            simple_network.forward(x),
            new_network.forward(x)
        )

class TestPyTorchConversion:
    """Test suite for PyTorch model conversion."""
    
    def test_layer_conversion(self):
        """Test individual layer conversion."""
        # Linear layer
        torch_linear = nn.Linear(10, 5)
        lrp_linear = convert_torch_layer(torch_linear)
        assert isinstance(lrp_linear, LRPDense)
        
        # ReLU
        torch_relu = nn.ReLU()
        lrp_relu = convert_torch_layer(torch_relu)
        assert isinstance(lrp_relu, LRPReLU)
        
        # Dropout
        torch_dropout = nn.Dropout(0.5)
        lrp_dropout = convert_torch_layer(torch_dropout)
        assert isinstance(lrp_dropout, LRPDropout)
        
        # BatchNorm
        torch_bn = nn.BatchNorm1d(10)
        lrp_bn = convert_torch_layer(torch_bn)
        assert isinstance(lrp_bn, LRPBatchNorm)

    def test_network_conversion(self):
        """Test full network conversion."""
        torch_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        lrp_network = convert_torch_to_lrp(
            torch_model,
            input_shape=(10,)
        )
        
        # Test forward pass equivalence
        x = torch.randn(1, 10)
        torch_out = torch_model(x).detach().numpy()
        lrp_out = lrp_network.forward(x.numpy())
        
        np.testing.assert_array_almost_equal(torch_out, lrp_out)

class TestUtilities:
    """Test suite for utility functions."""
    
    def test_sequential_lrp(self):
        """Test sequential network builder."""
        network = SequentialLRP(
            LRPDense(np.eye(3), np.zeros(3)),
            LRPReLU(),
            input_shape=(3,)
        )
        
        x = np.array([[1, -1, 2]])
        y = network.forward(x)
        assert y.shape == (1, 3)

    def test_create_classifier(self):
        """Test classifier creation utility."""
        network = create_classifier(
            input_dim=10,
            hidden_dims=[5, 3],
            num_classes=2,
            dropout=0.5
        )
        
        assert len(network.layers) == 8  # 3 dense, 2 relu, 2 dropout
        
        x = np.random.randn(1, 10)
        y = network.forward(x)
        assert y.shape == (1, 2)

def test_complex_network():
    """Integration test with complex network architecture."""
    # Create complex network
    network = LRPNetwork([
        LRPDense(np.random.randn(10, 8), np.zeros(8)),
        LRPBatchNorm(8),
        LRPReLU(),
        LRPDropout(0.3),
        LRPDense(np.random.randn(8, 4), np.zeros(4)),
        LRPReLU(),
        LRPDense(np.random.randn(4, 2), np.zeros(2))
    ], input_shape=(10,))
    
    # Test forward pass
    x = np.random.randn(32, 10)  # Batch of 32 samples
    y = network.forward(x, training=True)
    assert y.shape == (32, 2)
    
    # Test explanation
    relevance = network.explain(
        x[0:1],  # Single sample
        target_class=1,
        layer_names=['layer_0', 'layer_4']  # Get relevance for specific layers
    )
    
    # Verify relevance properties
    assert len(relevance) == 3  # Input + 2 requested layers
    for R in relevance.values():
        assert not np.any(np.isnan(R))
        assert not np.any(np.isinf(R))

if __name__ == "__main__":
    pytest.main([__file__])
