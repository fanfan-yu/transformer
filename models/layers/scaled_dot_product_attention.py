import numpy as np
import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_key):
        super(ScaledDotProductAttention, self).__init__()
        self.d_key = d_key

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_key)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
