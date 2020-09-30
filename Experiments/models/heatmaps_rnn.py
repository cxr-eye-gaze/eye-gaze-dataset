import torch
import torch.nn as nn
from .attention import Attention


class EncoderRNN(nn.Module):
    def __init__(self, input_size=300, cell='lstm', brnn=False, num_layers=3, rnn_hidden_dim=256, dropout=0.3, out_dim=64, attention=None):
        super().__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        num_directions = 2 if brnn else 1
        self.brnn = brnn
        self.rnn = getattr(nn, cell.upper())(
            input_size=input_size,
            hidden_size=rnn_hidden_dim // num_directions,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=brnn, batch_first=True)
        self.attention = Attention(query_dim=rnn_hidden_dim // num_directions) if attention else None
        self.dense = nn.Linear(rnn_hidden_dim, out_dim)

    def forward(self, x):
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden, cell = hidden
        if self.brnn:  # concat last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]  # last hidden
        if self.attention is not None:
            att_weights = self.attention(hidden, output, output)
            x = self.dense(att_weights)
        else:
            x = self.dense(output[:, -1, :])
        return x  #(batch_size, out_dim)