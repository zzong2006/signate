import os
import sys
import json
import logging

import torch
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class GameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, normalize=True, is_testset=False):
        if not is_testset:
            self.x = torch.tensor(df.drop(['blueWins'], axis=1).values, dtype=torch.float32)
            self.y = torch.tensor(df['blueWins'].values, dtype=torch.float32)
        else:
            self._ids = df['gameId'].values
            self.x = torch.tensor(df.values, dtype=torch.float32)
            self.y = None
            
        if normalize:
            if isinstance(normalize, tuple):
                self.mean, self.std = normalize
            else:
                self.mean = self.x.mean(0, True) # dim, dimkeep
                self.std = self.x.std(0, True)
            self.x = (self.x - self.mean) / self.std
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx, 1:], self.y[idx]
        else:
            return self.x[idx, 1:]


class DatasetHandler:
    def __init__(self, option_path='./config/dataset.json'):
        with open(option_path, "r") as jread:
            self.opt = json.load(jread)
        self.logger = logging.getLogger('DatatsetHandler') 
        self.logger.setLevel(logging.DEBUG)
        self.train_ds = GameDataset(self._read_data('train'), is_testset=False)
        self.test_ds = GameDataset(
            self._read_data('test'), 
            normalize=(self.train_ds.mean, self.train_ds.std),
            is_testset=True
        )
        
    def _read_data(self, target):
        """
            target: train or test
        """
        if not os.path.exists(self.opt[target]['pickle']):
            self.logger.info(f'cannot found pickle of ({target}). Load just csv')
            df = pd.read_csv(self.opt[target]['csv'])
            self.logger.info('convert csv to pickle')
            df.to_pickle(self.opt[target]['pickle'])
        else:
            self.logger.info(f'found pickle of ({target}). Load pickle')
            df = pd.read_pickle(self.opt[target]['pickle'])
        return df
    
    def _set_loader(self, dataset, shuffle, batch_size):
        """
            get torch DataLoader. 
            
            Default Setting
            * train 
                * shuffle: True
                * batch_size: 128
            * test 
                * shuffle: False
                * batch_size: 256
        """
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=16)
    
    def split(self):
        # use 20% of training data for validation
        game_dataset = self.train_ds
        train_ratio = 0.8
        train_set_size = int(len(game_dataset) * train_ratio)
        valid_set_size = len(game_dataset) - train_set_size

        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = torch.utils.data.random_split(game_dataset, [train_set_size, valid_set_size], generator=seed)
        return train_set, valid_set