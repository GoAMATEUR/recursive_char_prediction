import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

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


class CharTransformerDecoder(nn.Module):
    def __init__(self, input_size, output_size,  seq_len=128, d_model=512, hidden_dim=2048, dropout=0.1, num_layers=6, n_head=8):
        super(CharTransformerDecoder, self).__init__()
        # self.hidden_size = hidden_size
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=seq_len)
        self.input_emb = nn.Embedding(input_size, self.d_model)
        self.transformer_docoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model,
                                       nhead=n_head,
                                       dim_feedforward=hidden_dim,
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers, 
        )
        self.decoder = nn.Linear(d_model, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.init_weights()

    def forward(self, input):
        
        embedded_input = self.input_emb(input) * math.sqrt(self.d_model)
        embedded_input = self.pos_encoder(embedded_input) # (batch, seq_len, d_model)
        # tgt_mask = torch.triu(torch.ones(embedded_input.size(1), embedded_input.size(1), dtype=torch.bool)).to(input.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embedded_input.size(1)).to(input.device)
        # print(tgt_mask)
        # print(causal_mask)
        # causal_mask = torch.tril(torch.ones(embedded_input.size(1), embedded_input.size(1), dtype=torch.bool)).bool().to(input.device)
        output = self.transformer_docoder(embedded_input, embedded_input, tgt_mask)
        o1 = self.decoder(output)
        return o1

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)


class CharTransformer(nn.Module):
    def __init__(self, input_size, output_size,  seq_len=128, d_model=512, hidden_dim=2048, dropout=0.1, num_layers=6, n_head=8):
        super(CharTransformer, self).__init__()
        # self.hidden_size = hidden_size
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=seq_len)
        self.input_emb = nn.Embedding(input_size, self.d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.Linear(d_model, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.init_weights()

    def forward(self, input):
        
        embedded_input = self.input_emb(input)
        embedded_input = self.pos_encoder(embedded_input) # (batch, seq_len, d_model)
        # causal_mask = torch.triu(torch.ones(embedded_input.size(1), embedded_input.size(1), dtype=torch.bool), diagonal=1).to(input.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embedded_input.size(1)).to(input.device)
        # print(tgt_mask)
        # print(causal_mask)
        # causal_mask = torch.tril(torch.ones(embedded_input.size(1), embedded_input.size(1), dtype=torch.bool)).bool().to(input.device)
        output = self.transformer(embedded_input, embedded_input)
        o1 = self.decoder(output)
        return o1

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

# class CharPredictionTransformer(nn.Module):
#     def __init__(self, one_hot_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
#         super(CharPredictionTransformer, self).__init__()
        
#         # Transformer configuration
#         self.d_model = d_model
#         self.embedding = nn.Linear(one_hot_dim, d_model)  # Project one-hot vector to d_model dimension
#         self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))  # Positional encodings, assuming max seq_len of 5000
        
#         self.transformer = nn.Transformer(
#             d_model=d_model,
#             nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout
#         )
        
#         # Output layers for prediction
#         self.output_linear = nn.Linear(d_model, one_hot_dim)  # Project to output dimension
#         self.softmax = nn.Softmax(dim=-1)  # Softmax over character possibilities
        
#     def forward(self, x):
#         """
#         Forward pass for the character prediction transformer.
#         :param x: Input sequence of one-hot vectors, shape (batch, seq_len, one_hot_dim)
#         :return: Character probabilities for each position in the sequence
#         """
#         # Project one-hot input to d_model and add positional encoding
#         x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]

#         # Transformer expects input in shape (seq_len, batch, d_model)
#         x = x.permute(1, 0, 2)
        
#         # Transformer forward pass
#         transformer_output = self.transformer(x, x)
        
#         # Output linear layer and softmax for each time step
#         logits = self.output_linear(transformer_output)  # (seq_len, batch, one_hot_dim)
#         logits = logits.permute(1, 0, 2)  # (batch, seq_len, one_hot_dim)
        
#         # Return probability distribution over characters
#         return self.softmax(logits)

if __name__ == "__main__":
    transformer = CharTransformerDecoder(5, 5)
    x = torch.zeros(1, 2).long() # (batch_size, seq_length, input_size)
    
    output = transformer(x)
    print(output.shape)
    