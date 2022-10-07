import torch.nn as nn

from srl.model.layers.swish import Swish


class StateEncoder(nn.Module):

    def __init__(self, input_size, state_size, activation, dropout_rate, num_layers=1, bias=True):
        super(StateEncoder, self).__init__()
        _linears = []
        for i in range(num_layers):
            if i == 0:
                _linears.append(nn.Linear(input_size, state_size, bias=bias))
            else:
                _linears.append(nn.Linear(state_size, state_size, bias=bias))

        self.linears = nn.ModuleList(_linears)

        if activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = Swish()

        self.dropout = nn.Dropout(dropout_rate)
        self.output_size = state_size

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x
