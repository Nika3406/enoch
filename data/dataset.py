import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokens, block_size=256):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size])
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1])
        return x, y
