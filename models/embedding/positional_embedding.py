import numpy as np
import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self, dropout, max_len=5000, d_model=512):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        positional_array = np.array([[pos/ np.power(10000, 2*i/d_model) for i in range(d_model)]
                                         if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        # sin is used for even indices and cos is used for odd indices
        positional_array[:, 0::2] = np.sin(positional_array[:, 0::2])
        positional_array[:, 1::2] = np.cos(positional_array[:, 1::2])

        self.embedding = torch.FloatTensor(positional_array)

    def forward(self, inputs):
        # dropout and residual after positional embedding
        return self.dropout(inputs + self.embedding[:inputs.size(1), :])

