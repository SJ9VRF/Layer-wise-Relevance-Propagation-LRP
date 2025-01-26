
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
