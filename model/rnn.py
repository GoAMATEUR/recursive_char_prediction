import torch
import torch.nn as nn
# from utils.parse_data import CharParser

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_block = nn.RNN(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.h2o = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """
        input: (batch_size, seq_length, input_size)
        """
         # (a hidden layer, batch_size, hidden_size)
        output, hidden = self.rnn_block(input, hidden)
        output = self.h2o(output)
        return output, hidden


if __name__ == "__main__":
    rnn = RNN(10, 128, 20)
    x = torch.randn(8, 32, 10)
    rnn(x)

            