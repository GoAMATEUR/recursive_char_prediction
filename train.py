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
wandb.login()


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

train_dir = "./data/train"
test_dir = "./data/val"
full_dir = "./data/full"
batch_size = 64
seq_length = 32
hidden_size = 256
dropout = 0 # no dropout for 1-layer RNN
temperature = 1.0
log_interval = 1000

embedding_config = CharParser(full_dir)

dataset = CharDataset(train_dir, seq_length, embedding_config)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
testset = CharDataset(test_dir, seq_length, embedding_config)
testloader = DataLoader(testset, batch_size, shuffle=False)
print("initialized dataset")
rnn = RNN(dataset.vocab_size, hidden_size, dataset.vocab_size, dropout).to(device)

softmax_layer = nn.LogSoftmax(dim=-1)
criteria = nn.NLLLoss()
optimizer = Adam(rnn.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01, cooldown=2, min_lr=1e-6)

current_run_name = time.strftime("%Y-%m-%d-%H-%M") 
wandb.init(project="ESE5460_hw3_rnn", 
           name=current_run_name, 
           config={
                "batch_size": batch_size,
                "seq_length": seq_length,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "temperature": temperature,
                "log_interval": log_interval,
           })

output_dir = "output/{}".format(current_run_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_loss = AverageMeter()
for epoch in range(10):
    for i, (x, y) in enumerate(tqdm(dataloader)):
        rnn.train()
        hidden = torch.zeros(1, x.size(0), hidden_size).to(device)
        x, y = x.to(device), y.to(device) # (batch_size, seq_length, vocab_size), (batch_size, seq_length, 1)
        optimizer.zero_grad()
        output, _ = rnn(x, hidden) # (batch_size, seq_length, vocab_size)
        output = softmax_layer(output.view(-1, dataset.vocab_size) / temperature)
        loss = criteria(output, y.view(-1).long())
        # clip gradient
        loss.backward()
        total_loss.update(loss.item())
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        wandb.log({"loss/train": loss.item()})
        # wandb.log()
        if (i+1) % log_interval == 0:
            rnn.eval()
            # print(f"Epoch: {epoch}, Loss: {loss.item()}")
            print(f"ealuating at Epoch: {epoch}, Loss: {total_loss.avg}")
            wandb.log({"loss/train_avg": total_loss.avg, "perplexity/train": np.exp(total_loss.avg)})
            torch.save(rnn.state_dict(), os.path.join(output_dir, f"{epoch}_{i}_model.pth"))
            total_loss.reset()
            with torch.no_grad():
                total_test_loss = AverageMeter()
                for x, y in (tqdm(testloader)):
                    hidden = torch.zeros(1, x.size(0), hidden_size).to(device)
                    x, y = x.to(device), y.to(device)
                    output, _ = rnn(x, hidden)
                    output = softmax_layer(output.view(-1, dataset.vocab_size) / temperature)
                    loss = criteria(output, y.view(-1).long())
                    total_test_loss.update(loss.item())
                wandb.log({"loss/test": total_test_loss.avg, "perplexity/test": np.exp(total_test_loss.avg)})
            print("Test Loss: ", total_test_loss.avg)
torch.save(rnn.state_dict(), os.path.join(output_dir, f"final_model.pth"))
