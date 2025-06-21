# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
import os
import psutil
import pandas as pd

class SVDppDataset(Dataset):
    def __init__(self, df, user_map, item_map):
        self.users = df['user'].map(user_map).values
        self.items = df['item'].map(item_map).values
        self.ratings = df['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class SVDppModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.y_factors = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_bias.weight, 0, 0.01)
        nn.init.normal_(self.item_bias.weight, 0, 0.01)
        nn.init.normal_(self.user_factors.weight, 0, 0.01)
        nn.init.normal_(self.item_factors.weight, 0, 0.01)
        nn.init.normal_(self.y_factors.weight, 0, 0.01)

    def forward(self, u, i, Ru):
        # Ru: list of lists of item indices per batch element
        bu = self.user_bias(u).squeeze()
        bi = self.item_bias(i).squeeze()
        pu = self.user_factors(u)
        qi = self.item_factors(i)
        # implicit
        y_sum = torch.zeros_like(pu)
        for idx, ru in enumerate(Ru):
            if len(ru)>0:
                y_sum[idx] = self.y_factors(torch.tensor(ru, device=u.device)).mean(dim=0)
        pu_hat = pu + y_sum
        preds = bu + bi + (pu_hat * qi).sum(dim=1)
        return preds

class SVDpp(BaseModel):
    def __init__(self,
                 n_factors=20, lr=0.005, reg=0.02, n_epochs=50,
                 batch_size=1024, verbose=True, rating_scale=(0,100)):
        super().__init__()
        self.n_factors, self.lr, self.reg = n_factors, lr, reg
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.verbose = verbose
        self.rating_scale = rating_scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, train_df, val_df=None):
        # mappings
        self.user_map = {u:i for i,u in enumerate(train_df['user'].unique())}
        self.item_map = {j:i for i,j in enumerate(train_df['item'].unique())}
        n_users, n_items = len(self.user_map), len(self.item_map)
        # build datasets
        train_ds = SVDppDataset(train_df, self.user_map, self.item_map)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        if val_df is not None:
            val_ds = SVDppDataset(val_df, self.user_map, self.item_map)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        # implicit history
        user_pos = train_df.groupby('user')['item'].apply(lambda x: list(x.map(self.item_map))).to_dict()
        # model
        self.model = SVDppModel(n_users, n_items, self.n_factors).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        criterion = nn.MSELoss()
        # training
        for epoch in range(1, self.n_epochs+1):
            self.model.train()
            total_loss = 0.0
            for u,i,r in train_loader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                # prepare Ru for batch
                Ru = [user_pos[train_ds.users[idx]] for idx in u.cpu().numpy()]
                preds = self.model(u, i, Ru)
                loss = criterion(preds, r)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(r)
            train_rmse = math.sqrt(total_loss / len(train_ds))
            if val_df is not None:
                val_rmse = self._evaluate(val_loader)
                if self.verbose:
                    print(f"Epoch {epoch}/{self.n_epochs} - TrainRMSE: {train_rmse:.4f} - ValRMSE: {val_rmse:.4f}")
            else:
                if self.verbose:
                    print(f"Epoch {epoch}/{self.n_epochs} - TrainRMSE: {train_rmse:.4f}")
        self.training_time = 0  # omit

    def _evaluate(self, loader):
        self.model.eval()
        se, cnt = 0.0, 0
        with torch.no_grad():
            for u,i,r in loader:
                u,i,r = u.to(self.device), i.to(self.device), r.to(self.device)
                Ru = [[] for _ in range(len(u))]  # skip implicit in val
                preds = self.model(u, i, Ru)
                se += ((preds - r)**2).sum().item()
                cnt += len(r)
        return math.sqrt(se/cnt) if cnt>0 else float('nan')

    def predict(self, df):
        ds = SVDppDataset(df, self.user_map, self.item_map)
        loader = DataLoader(ds, batch_size=self.batch_size)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for u,i,_ in loader:
                u,i = u.to(self.device), i.to(self.device)
                preds.extend(self.model(u, i, [[]]*len(u)).cpu().numpy())
        # clip
        return [max(self.rating_scale[0], min(self.rating_scale[1], p)) for p in preds]
