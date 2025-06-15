import torch
from torch import nn

from models.embedding.positional_embedding import PositionalEmbedding
from models.layers.decoder_layer import DecoderLayer
from models.utils.util import get_attn_pad_mask, get_attn_subsequence_mask


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_key, d_value, d_feedforward, tgt_vocab_size, max_len, num_decoder_layers, dropout=0.1):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(dropout, max_len, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_key, d_value, d_feedforward, dropout) for _ in range(num_decoder_layers)
        ])

    def forward(self, inputs, enc_inputs, enc_outputs):
        outputs = self.embedding(inputs)
        outputs = self.pos_embedding(outputs)

        attn_pad_mask = get_attn_pad_mask(inputs, inputs)
        attn_subsequence_mask = get_attn_subsequence_mask(inputs)
        attn_mask = torch.gt((attn_pad_mask + attn_subsequence_mask), 0)
        enc_attn_mask = get_attn_pad_mask(inputs, enc_inputs)

        attns, enc_attns = [], []
        for layer in self.layers:
            outputs, attn, enc_attn = layer(outputs, enc_outputs, attn_mask, enc_attn_mask)
            attns.append(attn)
            enc_attns.append(enc_attn)

        return outputs, attns, enc_attns
