import torch
import torch.nn as nn
from model.rnn import RNN
from utils.dataset import CharDataset
from utils.avgmeter import AverageMeter
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import time
import wandb
wandb.login()


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

data_dir = "./data"
batch_size = 128
seq_length = 32
hidden_size = 128
temperature = 2.0
log_interval = 100

dataset = CharDataset("data", seq_length)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

rnn = RNN(dataset.vocab_size, hidden_size, dataset.vocab_size).to(device)

softmax_layer = nn.LogSoftmax(dim=-1)
criteria = nn.NLLLoss()
optimizer = Adam(rnn.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)


current_run_name = time.strftime("%Y-%m-%d-%H-%M") 
wandb.init(project="ESE5460_hw3_1", name=current_run_name, config={"batch_size": batch_size, "seq_length": seq_length, "hidden_size": hidden_size, "temperature": temperature})

total_loss = AverageMeter()
for epoch in range(1000):
    for i, (x, y) in enumerate(tqdm(dataloader)):
        hidden = torch.zeros(1, batch_size, hidden_size).to(device)
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
        if i % 1000 == 0:
            # print(f"Epoch: {epoch}, Loss: {loss.item()}")
            wandb.log({"loss/train_avg": total_loss.avg})
            torch.save(rnn.state_dict(), f"output/{epoch}_{i}_model.pth")
            total_loss.reset()
torch.save(rnn.state_dict(), f"output/final_model.pth")
