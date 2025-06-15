from torch import nn
from models.embedding.positional_embedding import PositionalEmbedding
from models.layers.encoder_layer import EncoderLayer
from models.utils.util import get_attn_pad_mask


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_key, d_value, d_feedforward, d_embed, max_len, num_encoder_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(d_embed, d_model)
        self.pos_embedding = PositionalEmbedding(dropout, max_len, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_key, d_value, d_feedforward, dropout) for _ in range(num_encoder_layers)
        ])

    def forward(self, inputs):
        # inputs: [batch_size, seq_len]
        outputs = self.embedding(inputs)
        outputs = self.pos_embedding(outputs)

        attn_mask = get_attn_pad_mask(inputs, inputs)
        self_attns = []

        for layer in self.layers:
            outputs, self_attn = layer(outputs, attn_mask)
            self_attns.append(self_attn)
        return outputs, self_attns