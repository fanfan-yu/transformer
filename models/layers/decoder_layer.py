from torch import nn
from models.layers.feed_forward import FeedForward
from models.layers.multi_head_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_key, d_value, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.mask_attn = MultiHeadAttention(d_model, n_head, d_key, d_value)
        self.attn = MultiHeadAttention(d_model, n_head, d_key, d_value)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, inputs, enc_outputs, decoder_mask, encoder_mask):
        outputs, attn = self.mask_attn(inputs, inputs, inputs, decoder_mask)
        outputs, enc_attn = self.attn(outputs, enc_outputs, enc_outputs, encoder_mask)
        outputs = self.feed_forward(outputs)
        return outputs, attn, enc_attn
