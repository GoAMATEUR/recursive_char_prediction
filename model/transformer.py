import torch
import torch.nn as nn


class transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(transformer, self).__init__()
        self.hidden_size = hidden_size
        self.transformer_block = nn.Transformer(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=-1)