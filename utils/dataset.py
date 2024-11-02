from torch.utils.data import Dataset, DataLoader
import torch
import os
import json


class CharParser:
    
        def __init__(self, data_path):
            self.data_path = data_path
            self.vocab = None
            self.vocab_size = None
            self.vocab_to_id = None
            self.id_to_vocab = None
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
            data = self.read_data_recursive()
            self.vocab = sorted(set(data))
            self.vocab_size = len(self.vocab)
            self.vocab_to_id = {char: idx for idx, char in enumerate(self.vocab)}
            self.id_to_vocab = {idx: char for char, idx in self.vocab_to_id.items()}
            print("Vocab size: ", self.vocab_size)
    
        
    
        def embedding_seq_to_char(self, embedding: torch.Tensor) -> str:
            # embedding: (batch_size, seq_length, vocab_size)
            output = [""] * embedding.size(0)
            max_indices = torch.argmax(embedding, dim=-1)
            for i in range(embedding.size(0)):
                for j in range(embedding.size(1)):
                    output[i] += self.id_to_char(max_indices[i, j].item())
            return output

        def indices_to_chars(self, indices: torch.Tensor) -> str:
            # indices: (batch_size, seq_length)
            # given a tensor of batches of indices, convert to string
            output = [""] * indices.size(0)
            for i in range(indices.size(0)):
                for j in range(indices.size(1)):
                    output[i] += self.id_to_char(indices[i, j].item())
            return output
    
        def id_to_char(self, idx: int) -> str:
            # single id to char
            return self.id_to_vocab[idx]
        
        def char_to_embedding(self, char: str) -> int:
            # single char to embedding
            embedding = torch.zeros(self.vocab_size, dtype=torch.long)
            embedding[self.vocab_to_id[char]] = 1.0
            return embedding

        def chars_to_ids(self, chars: str) -> torch.Tensor:
            # chars: str
            # given a sentence of characters, convert to tensor of indices
            output = torch.zeros(1, len(chars), dtype=torch.float32) # (1, seq_length)
            for i in range(len(chars)):
                output[0, i] = self.vocab_to_id[chars[i]]
            return output

class CharDataset(Dataset):

    def __init__(self, data_path: str, seq_length: int, embedding_config: CharParser, use_embedding_layer: bool=False):
        self.data_path = data_path
        self.seq_length = seq_length
        self.embedding_config = embedding_config
        self.vocab_size = embedding_config.vocab_size
        self.data = self.read_data_recursive()
        self.use_embedding_layer = use_embedding_layer
    
    def read_data_recursive(self) -> str:
        data = ""
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r") as f:
                        data += f.read()
        return data

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        if not self.use_embedding_layer:
            x = torch.zeros(self.seq_length, self.vocab_size, dtype=torch.float32)
            y = torch.zeros(self.seq_length, dtype=torch.long)
            x_str = self.data[idx:idx+self.seq_length]
            y_str = self.data[idx+1:idx+self.seq_length+1]
            for i, (x_char, y_char) in enumerate(zip(x_str, y_str)):
                x[i, self.embedding_config.vocab_to_id[x_char]] = 1.0
                y[i] = self.embedding_config.vocab_to_id[y_char]
        else:
            x = torch.zeros(self.seq_length, dtype=torch.long)
            y = torch.zeros(self.seq_length, dtype=torch.long)
            x_str = self.data[idx:idx+self.seq_length]
            y_str = self.data[idx+1:idx+self.seq_length+1]
            for i, (x_char, y_char) in enumerate(zip(x_str, y_str)):
                x[i] = self.embedding_config.vocab_to_id[x_char]
                y[i] = self.embedding_config.vocab_to_id[y_char]
        return x, y

    # def load_embedding(self, embedding_path: str):
    #     with open(embedding_path, "r") as file:
    #         self.vocab_to_id = json.loads(file.read())
    #     self.vocab_size = len(self.vocab_to_id)

    def char_to_embedding(self, char: str) -> int:
        embedding = torch.zeros(self.vocab_size, dtype=torch.float32)
        embedding[self.embedding_config.vocab_to_id[char]] = 1.0
        return embedding

    def embedding_to_char(self, embedding: torch.Tensor) -> str:
        ...

    def id_to_char(self, idx: int) -> str:
        return self.embedding_config.id_to_vocab[idx]


if __name__ == "__main__":
    data_path = "data"
    dataset = CharDataset(data_path)
    x, y = dataset[0]
    print(x.shape, y.shape)