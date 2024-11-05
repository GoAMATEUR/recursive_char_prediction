import torch
import torch.nn as nn
from model.transformer import CharTransformer, CharTransformerDecoder
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
import json

wandb_log = True
eval_loss = True
eval_generation = True

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

train_dir = "./data/train"
test_dir = "./data/val"
full_dir = "./data/full"
batch_size = 128
# seq_length = 128
d_model = 256
hidden_size =1024
dropout = 0.1
seq_length = 128
n_layers = 6
n_head = 8
# temperature = 1.0
log_interval = 1000
lr = 1e-5

embedding_config = CharParser(full_dir)


model = CharTransformerDecoder(embedding_config.vocab_size, embedding_config.vocab_size, 
                        seq_len=seq_length,
                        d_model=d_model,
                        n_head=n_head,
                        num_layers=n_layers,
                        hidden_dim=hidden_size,
                        dropout=dropout).to(device)


model.load_state_dict(torch.load("./output/transformer/2024-11-04-03-37/best_model.pth"))

model.eval()
input_text = "The formula one is held in brazil this week"
generated_text = "".join(input_text) 

with torch.no_grad():   
# Perform random generation from random starting point
    for i in range(128):
        input = embedding_config.chars_to_ids(input_text).to(device) # (1, seq_len)
        output= model(input) # (1, seq_len, vocab_size)
        # output = output[:, -1, :]
        last_output = output[:, -1, :]  # Get output at the last time step, shape: [1, vocab_size]
        probabilities = torch.softmax(last_output, dim=-1)
        # print("Probabilities: ", probabilities)
        predicted_index = torch.multinomial(probabilities, num_samples=1).item()
        next_char = embedding_config.id_to_char(predicted_index)
        # print("Next char: ", next_char)
        
        generated_text += next_char
        input_text = generated_text[-seq_length:]
print("Generated text: ", generated_text)

