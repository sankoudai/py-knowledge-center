import torch
from torch.utils.data import Dataset

class ExampleTensorDataset(Dataset):
    def __init__(self, tensor):
        super(ExampleTensorDataset, self).__init__()
        self.tensor = tensor

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.shape[0]