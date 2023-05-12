import mlconfig
import pandas as pd
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, transform, file, train=True):
        super(Dataset, self).__init__()
        self.transform = transform
        self.file = file
        self.train = train
        self.df = pd.read_csv(self.file).to_numpy()
#         self.df = pd.read_csv(self.file)
#         if train is True:
#             self.df = pd.concat([self.df[self.df['EF']<50], self.df[self.df['EF']>=50].sample(n=2000)]).to_numpy()
#         else:
#             self.df = self.df.to_numpy()
        self.length = len(self.df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pid, uid, ef, bef = self.df[index]
        data = pd.read_csv(f'./data/ecg/{uid}.csv').to_numpy()
#         if self.train is True and index+1 == self.length:
#             df = pd.read_csv(self.file)
#             self.df = pd.concat([df[df['EF']<50], df[df['EF']>=50].sample(n=2000)]).to_numpy()
        return self.transform(data).type(torch.FloatTensor), bef, ef
        
        

@mlconfig.register
class DataLoader(data.DataLoader):

    def __init__(self, file: str, train: bool, batch_size: int, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        super(DataLoader, self).__init__(dataset=Dataset(transform, file, train), batch_size=batch_size, shuffle=train, **kwargs)
