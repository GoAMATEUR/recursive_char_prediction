from torch.utils.data import Dataset, DataLoader
import torch
import os
import json


class CharDataset(Dataset):

    def __init__(self, data_path: str, seq_length: int=32):
        self.data_path = data_path
        self.seq_length = seq_length
        self.vocab_to_id = None
        self.vocab_size = None
        self.data = None
        self.init_embedding()
    
    def read_data_recursive(self) -> str:
        data = ""
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r") as f:
                        data += f.read()
        return data

    def init_embedding(self) -> set:
        self.data = self.read_data_recursive()
        vocab = set(self.data)
        self.vocab_size = len(vocab)
        self.vocab_to_id = {char: idx for idx, char in enumerate(vocab)}
        self.id_to_vocab = {idx: char for char, idx in self.vocab_to_id.items()}
        print("Vocab size: ", self.vocab_size)

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        x = torch.zeros(self.seq_length, self.vocab_size, dtype=torch.float32)
        y = torch.zeros(self.seq_length, dtype=torch.long)
        x_str = self.data[idx:idx+self.seq_length]
        y_str = self.data[idx+1:idx+self.seq_length+1]
        for i, (x_char, y_char) in enumerate(zip(x_str, y_str)):
            x[i, self.vocab_to_id[x_char]] = 1.0
            y[i] = self.vocab_to_id[y_char]
        return x, y

    def load_embedding(self, embedding_path: str):
        with open(embedding_path, "r") as file:
            self.vocab_to_id = json.loads(file.read())
        self.vocab_size = len(self.vocab_to_id)

    def char_to_embedding(self, char: str) -> int:
        embedding = torch.zeros(self.vocab_size, dtype=torch.float32)
        embedding[self.vocab_to_id[char]] = 1.0
        return embedding

    def embedding_to_char(self, embedding: torch.Tensor) -> str:
        ...

    def id_to_char(self, idx: int) -> str:
        return self.id_to_vocab[idx]


if __name__ == "__main__":
    data_path = "data"
    dataset = CharDataset(data_path)
    x, y = dataset[0]
    print(x.shape, y.shape)