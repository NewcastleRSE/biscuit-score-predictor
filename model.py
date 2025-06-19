import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        # Very simple model due to small training dataset
        self.linear = nn.Linear(in_size, 1)

    def forward(self,x):
        return self.linear(x)
    