from torch import nn
from models.layers.feed_forward import FeedForward
from models.layers.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_key, d_value, d_feedforward=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, d_key, d_value)
        self.feed_forward = FeedForward(d_model, d_feedforward, dropout=dropout)

    def forward(self, inputs, mask):
        enc_outputs, attn = self.attn(inputs, inputs, inputs, mask)
        enc_outputs = self.feed_forward(enc_outputs)
        return enc_outputs, attn
