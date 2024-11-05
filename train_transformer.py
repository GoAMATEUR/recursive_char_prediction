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

dataset = CharDataset(train_dir, seq_length, embedding_config, use_embedding_layer=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
print("len of dataset: ", len(dataset))
testset = CharDataset(test_dir, seq_length, embedding_config, use_embedding_layer=True)
testloader = DataLoader(testset, batch_size, shuffle=False)
print("len of testset: ", len(testset))
print("initialized dataset")
model = CharTransformerDecoder(embedding_config.vocab_size, embedding_config.vocab_size, 
                        seq_len=seq_length,
                        d_model=d_model,
                        n_head=n_head,
                        num_layers=n_layers,
                        hidden_dim=hidden_size,
                        dropout=dropout).to(device)

# softmax_layer = nn.LogSoftmax(dim=-1)
criteria = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, threshold=0.01, cooldown=2, min_lr=1e-8)


current_run_name = time.strftime("decoder_128_nosche_%Y-%m-%d-%H-%M") 
if wandb_log:
    wandb.login()
    wandb.init(project="ESE5460_hw3_tf_train", 
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
for epoch in range(2000):
    for i, (x, y) in enumerate(tqdm(dataloader)):
        
        x, y = x.to(device), y.to(device) # (batch_size, seq_length), (batch_size, seq_length)
        optimizer.zero_grad()
        output = model(x) # (batch_size, seq_length, vocab_size)
        # print(embedding_config.indices_to_chars(y), embedding_config.embedding_seq_to_char(output))
        # print("Sample output: ", embedding_config.embedding_seq_to_char(output[:2]), embedding_config.indices_to_chars(y[:2]))
        # print(output)
        acc = (torch.argmax(output, dim=-1) == y).sum().item() / y.shape[0] / y.shape[1]
        output = output.view(-1, dataset.vocab_size)
        # print(output)÷
        loss = criteria(output, y.view(-1).long())
        # print(loss)
        loss.backward()
        total_loss.update(loss.item())
        optimizer.step()
        step_counter += 1
        # print(acc)
        if wandb_log:
            wandb.log({"loss/train": loss.item()})
            wandb.log({"acc/train": acc})
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
                    acc_totoal = 0
                    for x, y in tqdm(testloader):
                        # hidden = torch.zeros(1, x.size(0), hidden_size).to(device)
                        x, y = x.to(device), y.to(device)
                        output = model(x)
                        output_softmax = output.view(-1, dataset.vocab_size)
                        # calculate accuracy
                        acc = (torch.argmax(output_softmax, dim=-1) == y.view(-1)).sum().item() / y.view(-1).shape[0]
                        acc_totoal += acc
                        loss = criteria(output_softmax, y.view(-1).long())
                        total_test_loss.update(loss.item())
                        # print(output.shape)
                    if wandb_log:
                        wandb.log({"loss/validation": total_test_loss.avg, "perplexity/test": np.exp(total_test_loss.avg), "acc/validation": acc_totoal / len(testloader)})
                        
                    print("Sample output: ", embedding_config.embedding_seq_to_char(output[:2]), embedding_config.indices_to_chars(y[:2]))
                    if total_test_loss.avg < best_loss:
                        best_loss = total_test_loss.avg
                        torch.save(model.state_dict(), os.path.join(output_dir, f"best_model.pth"))
                        print("New best test Loss: ", total_test_loss.avg)
                    # scheduler.step(total_test_loss.avg)
                    optimizer_lr = optimizer.param_groups[0]['lr']
                    if wandb_log:
                        wandb.log({"learning_rate": optimizer_lr})
            if eval_generation:
                model.eval()
                generated_text = "I am a stud"
                with torch.no_grad():   
                # Perform random generation from random starting point
                    for i in range(seq_length - len(generated_text)):
                        input = embedding_config.chars_to_ids(generated_text).to(device) # (1, seq_len)
                        output= model(input) # (1, seq_len, vocab_size)
                        # output = output[:, -1, :]
                        last_output = output[:, -1, :]  # Get output at the last time step, shape: [1, vocab_size]
                        probabilities = torch.softmax(last_output.squeeze(0), dim=0)
                        predicted_index = torch.multinomial(probabilities, num_samples=1).item()
                        next_char = embedding_config.id_to_char(predicted_index)
                        # print("Next char: ", next_char) 
                        generated_text += next_char
                print("Generated text: ", generated_text)
            model.train()
        
    dataset.reset_index_offset()
torch.save(model.state_dict(), os.path.join(output_dir, f"final_model.pth"))
