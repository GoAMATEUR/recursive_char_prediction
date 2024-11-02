import torch
import torch.nn as nn
from model.rnn import RNN
from utils.dataset import CharDataset
from utils.dataset import CharParser
from utils.avgmeter import AverageMeter
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import time
import wandb
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

full_dir = "./data/full"
batch_size = 1
seq_length = 32
hidden_size = 256
dropout = 0 # no dropout for 1-layer RNN
temperature = 1.0
log_interval = 1000

embedding_config = CharParser(full_dir)

rnn = RNN(embedding_config.vocab_size, hidden_size, embedding_config.vocab_size, dropout).to(device)
rnn.load_state_dict(torch.load("output/2024-11-02-04-32/1_2999_model.pth"))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01, cooldown=2, min_lr=1e-6)
rnn.eval()


hidden = torch.zeros(1, batch_size, hidden_size).to(device)
x_t = "h"
x_t = CharParser.char_to_embedding(x_t)
x_t = x_t.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, vocab_size)
output = "h"
for i in range(seq_length - 1):
    # Recursively generate text until the desired length is reached
    output, hidden = rnn(x_t, hidden)
    max_index = torch.argmax(output, dim=-1)
    new_char = embedding_config.id_to_char(max_index.item())
    x_t = embedding_config.char_to_embedding(new_char).unsqueeze(0).unsqueeze(0).to(device)
    # x_t = x_t
    output += new_char

print(output)
    