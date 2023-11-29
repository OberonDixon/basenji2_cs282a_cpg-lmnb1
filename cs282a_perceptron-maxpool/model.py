from collections import OrderedDict
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MLPModel, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential(OrderedDict([
            ('conv1x1', nn.Conv1d(1536, 500, 1)),
            ('gelu1', nn.GELU()),
            ('maxpool1', nn.MaxPool1d(896)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(500, 18))
        ]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x