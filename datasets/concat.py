import torch

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_f, datasets_g):
        self.datasets_f = datasets_f
        self.datasets_g = datasets_g

    def __getitem__(self, i):
        return {
            'data_f' : self.datasets_f[i],
            'data_g' : self.datasets_g[i]
        }
        
    def __len__(self):
        return min(len(self.datasets_f), len(self.datasets_g))