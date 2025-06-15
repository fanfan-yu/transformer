from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, d_key, d_value, d_feedforward, d_embed, max_len, num_encoder_layers, num_decoder_layers, tgt_vocab_size, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, n_head, d_key, d_value, d_feedforward, d_embed, max_len, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, n_head, d_key, d_value, d_feedforward, tgt_vocab_size, max_len, num_decoder_layers, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_attns = self.encoder(enc_inputs)

        dec_outputs, dec_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.linear(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1))

