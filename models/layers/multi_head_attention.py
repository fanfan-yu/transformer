from torch import nn
from models.layers.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_key, d_value):
        super(MultiHeadAttention, self).__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.d_model = d_model

        # calculate weight matrix
        self.weight_query = nn.Linear(d_model, d_key * n_head, bias=False)
        self.weight_key = nn.Linear(d_model, d_key * n_head, bias=False)
        self.weight_value = nn.Linear(d_model, d_value * n_head, bias=False)
        self.weight_func = nn.Linear(n_head * d_value, d_model, bias=False)
        self.scale_dot_product_attention = ScaledDotProductAttention(d_key)

    # key: [batch_size, len_k, d_model]
    # value: [batch_size, len_v(=len_k), d_model]
    # query: [batch_size, len_q, d_model]
    # attn_mask: [batch_size, len_seq, len_seq]
    def forward(self, input_query, input_key, input_value, attn_mask):
        residual, batch_size = input_query, input_query.size(0)
        # calculate Q, K, V
        Q = self.weight_query(input_query).view(batch_size, -1, self.n_head, self.d_key).transpose(1, 2)
        K = self.weight_key(input_key).view(batch_size, -1, self.n_head, self.d_key).transpose(1, 2)
        V = self.weight_value(input_value).view(batch_size, -1, self.n_head, self.d_value).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attn = self.scale_dot_product_attention(Q, K, V, attn_mask)

        context = context.transpose(1, 2).reshape(input_query.size(0), -1, self.n_head * self.d_value)

        output = self.weight_func(context)
        return nn.LayerNorm(self.d_model)(output + residual), attn