import torch
import torch.nn as nn


class transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=512):
        super(transformer, self).__init__()
        # self.hidden_size = hidden_size
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(input_size, d_model)
        self.transformer_block = nn.Transformer(d_model=d_model, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        # self.h2o = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=-1)
        


    def forward(self, input):
        output = self.transformer_block(input)
        return


if __name__ == "__main__":
    transformer = transformer(10, 20)
    x = torch.randn(8, 32, 10) # (batch_size, seq_length, input_size)
    output = transformer(x)