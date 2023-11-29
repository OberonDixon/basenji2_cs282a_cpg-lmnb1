"""Applies a 1D convolution on the embeddings after the 7 convolutional blocks
and 11 dilated residual blocks of Basenji2.

This is for the linear probing part of the project, replicating the final Dense layer
of Basenji2 for data on nuclear lamination and CpG methylation. The aspects of the
Dense layer such as initialization and regularization can be found below:

https://colab.research.google.com/drive/14dDJBEdOZNBUu9a--wknuUULz5N3M65Q?usp=sharing

Author: Jimin Jung
"""

from torch import nn


class LinearTransform(nn.Module):
    """Takes in input (B, 1536, 896) and outputs predictions (B, 18, 896)."""

    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Conv1d(in_channels=1536, out_channels=18, kernel_size=1)
        nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv_layer.bias)
        self.activation = nn.Softplus()

    def forward(self, x):
        out = None
        out = self.activation(self.conv_layer(x))
        return out
