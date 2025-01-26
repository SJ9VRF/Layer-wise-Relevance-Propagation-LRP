# Layer-wise Relevance Propagation (LRP)

A PyTorch-compatible implementation of Layer-wise Relevance Propagation for neural network interpretability.

![Screenshot_2025-01-26_at_4 09 04_AM-removebg-preview](https://github.com/user-attachments/assets/f5926d91-7bee-4750-b30d-03ee1f87299c)



## Features
- LRP-ε rule implementation for dense layers
- ReLU activation support
- PyTorch model conversion utility
- Conservation property verification

## Installation
```bash
git clone https://github.com/username/lrp-pytorch.git
cd lrp-pytorch
pip install -r requirements.txt
```

## Requirements
- numpy
- torch

## Usage
```python
import torch.nn as nn
from lrp import convert_torch_to_lrp

# Create PyTorch model
model = nn.Sequential(
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 2)
)

# Convert to LRP network
lrp_net = convert_torch_to_lrp(model)

# Get explanations
x = np.random.randn(4)
relevance = lrp_net.explain(x, target_class=0)
```
