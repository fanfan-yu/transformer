from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.func = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model),
        )

    def forward(self, inputs):
        residual = inputs
        outputs = self.func(inputs)
        return nn.LayerNorm(self.d_model)(outputs + residual)
