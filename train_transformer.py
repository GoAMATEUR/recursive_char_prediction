import torch
import torch.nn as nn
from model.transformer import CharTransformer
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
batch_size = 32
# seq_length = 128
d_model = 200
hidden_size = 256
dropout = 0.1
seq_length = 128
n_layers = 3
n_head = 4
# temperature = 1.0
log_interval = 1000
lr = 0.01

embedding_config = CharParser(full_dir)

dataset = CharDataset(train_dir, seq_length, embedding_config, use_embedding_layer=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
print("len of dataset: ", len(dataset))
testset = CharDataset(test_dir, seq_length, embedding_config, use_embedding_layer=True)
testloader = DataLoader(testset, batch_size, shuffle=False)
print("len of testset: ", len(testset))
print("initialized dataset")
model = CharTransformer(embedding_config.vocab_size, embedding_config.vocab_size, 
                        seq_len=seq_length,
                        d_model=d_model,
                        n_head=n_head,
                        num_layers=n_layers,
                        hidden_dim=hidden_size,
                        dropout=dropout).to(device)

softmax_layer = nn.LogSoftmax(dim=-1)
criteria = nn.NLLLoss()
optimizer = Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.01, cooldown=2, min_lr=1e-6)


current_run_name = time.strftime("%Y-%m-%d-%H-%M") 
if wandb_log:
    wandb.login()
    wandb.init(project="ESE5460_hw3_tf", 
            name=current_run_name, 
            config={
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "hidden_size": hidden_size,
                    "d_model": d_model,
                    "dropout": dropout,
                    "n_layers": n_layers,
                    # "temperature": temperature,
                    "log_interval": log_interval,
                    "char_to_idx": embedding_config.vocab_to_id,
            })

output_dir = "output/transformer/{}".format(current_run_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

best_loss = float("inf")

total_loss = AverageMeter()
model.train()
step_counter = 0
for epoch in range(2):
    for i, (x, y) in enumerate(tqdm(dataloader)):
        
        x, y = x.to(device), y.to(device) # (batch_size, seq_length), (batch_size, seq_length)
        optimizer.zero_grad()
        output = model(x) # (batch_size, seq_length, vocab_size)
        # print(embedding_config.indices_to_chars(y), embedding_config.embedding_seq_to_char(output))
        # print("Sample output: ", embedding_config.embedding_seq_to_char(output[:2]), embedding_config.indices_to_chars(y[:2]))
        # print(output)
        output = softmax_layer(output.view(-1, dataset.vocab_size))
        # print(output)÷
        loss = criteria(output, y.view(-1).long())
        # print(loss)
        loss.backward()
        total_loss.update(loss.item())
        optimizer.step()
        step_counter += 1
        scheduler.step(loss.item())
        if wandb_log:
            wandb.log({"loss/train": loss.item()})
        # wandb.log()
        if step_counter % log_interval == 0:
            
            # print(f"Epoch: {epoch}, Loss: {loss.item()}")
            print(f"ealuating at Epoch: {epoch}, Loss: {total_loss.avg}")
            if wandb_log:
                wandb.log({"loss/train_avg": total_loss.avg, "perplexity/train": np.exp(total_loss.avg)})
            torch.save(model.state_dict(), os.path.join(output_dir, f"{epoch}_{i}_{step_counter}_model.pth"))
            total_loss.reset()
            
            if eval_loss:
                model.eval()
                with torch.no_grad():
                    total_test_loss = AverageMeter()
                    for x, y in (tqdm(testloader)):
                        # hidden = torch.zeros(1, x.size(0), hidden_size).to(device)
                        x, y = x.to(device), y.to(device)
                        output = model(x)
                        output_softmax = softmax_layer(output.view(-1, dataset.vocab_size))
                        loss = criteria(output_softmax, y.view(-1).long())
                        total_test_loss.update(loss.item())
                        # print(output.shape)
                    if wandb_log:
                        wandb.log({"loss/validation": total_test_loss.avg, "perplexity/test": np.exp(total_test_loss.avg)})
                        
                    print("Sample output: ", embedding_config.embedding_seq_to_char(output[:2]), embedding_config.indices_to_chars(y[:2]))
                    if total_test_loss.avg < best_loss:
                        best_loss = total_test_loss.avg
                        torch.save(model.state_dict(), os.path.join(output_dir, f"best_model.pth"))
                        print("New best test Loss: ", total_test_loss.avg)
            if eval_generation:
                model.eval()
                with torch.no_grad():   
                # Perform random generation from random starting point
                    generated_text = "happy"
                    for i in range(seq_length - len(generated_text)):
                        input = embedding_config.chars_to_ids(generated_text).to(device) # (1, seq_len)
                        output= model(input) # (1, seq_len, vocab_size)
                        # print("Output shape: ", output.shape) 
                        max_index = torch.argmax(output[0, -1, :], dim=-1) 
                        next_char = embedding_config.id_to_char(max_index.item())
                        # print("Next char: ", next_char) 
                        generated_text += next_char
                print("Generated text: ", generated_text)
            model.train()
    dataset.reset_index_offset()
torch.save(model.state_dict(), os.path.join(output_dir, f"final_model.pth"))
