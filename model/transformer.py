import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class transformer(nn.Module):
    def __init__(self, input_size, output_size,  seq_len=128, d_model=512, hidden_dim=2048, dropout=0.1):
        super(transformer, self).__init__()
        # self.hidden_size = hidden_size
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=seq_len)
        self.input_emb = nn.Embedding(input_size, self.d_model)
        self.transformer_block = nn.Transformer(d_model=self.d_model, 
                                                nhead=8, 
                                                dim_feedforward=hidden_dim, 
                                                dropout=dropout,
                                                batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        embedding = self.input_emb(input)
        embedding = self.pos_encoder(embedding)
        print("embedding shape: ", embedding.shape)
        tqt = torch.zeros_like(embedding)
        output = self.transformer_block(embedding, tqt)
        return output

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

if __name__ == "__main__":
    transformer = transformer(10, 20)
    x = torch.zeros(8, 32).long() # (batch_size, seq_length, input_size)
    
    output = transformer(x)
    print(output.shape)
    