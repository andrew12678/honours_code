import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from scipy import stats
from scipy.io import mmread
from glob import glob
import re
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from captum import attr
from scipy.stats import rankdata
import pandas as pd

class MatrixDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx,:]), torch.tensor(self.target[idx])

class Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        h_relu = self.act(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred
    

class Trainer:
    def __init__(self, model, optim, criterion, train_loader, val_loader, epoch, device, save_path, running=20):
        self.optim = optim
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epoch = epoch
        self.device = device
        self.running = running
        self.model = model.to(self.device)
        self.save_path = save_path

        print(self.save_path)

    def train_epoch(self):
        # Commented out because we don't use batch norms
        self.model.train()
        total = 0
        running_total = 0
        train_loss = 0
        train_loss_running = 0
        train_acc = 0
        train_running_acc = 0
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                  desc=f'Epoch: {self.curr_epoch}/{self.max_epoch} - Train', position=0, leave=True) as t:
            for batch_idx, (data, target) in t:
                data, target = data.to(self.device), target.to(self.device)
                self.optim.zero_grad()
                output = self.model(data)
                with torch.no_grad():
                    pred = torch.argmax(output, dim=1)
                    correct = torch.sum(pred == target).item()
                    train_acc += correct
                    train_running_acc += correct
                loss = self.criterion(output, target)
                loss_data = loss.item()

                total += len(data)
                train_loss += loss_data
                running_total += len(data)
                train_loss_running += loss_data

                if (batch_idx + 1) % self.running == 0:
                    t.set_postfix({'loss': train_loss_running / running_total, 'acc': train_running_acc / running_total})
                    running_total = 0
                    train_loss_running = 0
                    train_running_acc = 0

                loss.backward()
                self.optim.step()
        train_metrics = {'train_loss': train_loss / total, 'train_acc': train_acc / total}
        val_metrics = self.validate()
        for k,v in val_metrics.items():
            train_metrics[k] = v
        return train_metrics

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(1, self.max_epoch+1):
            self.curr_epoch = epoch
            metrics = self.train_epoch()
            metrics["epoch"] = epoch
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
            metrics_dp = {k : float('{:.4f}'.format(v)) for k,v in metrics.items()}
            with open(f'{self.save_path}/log.txt', 'a') as f:
                f.write(str(metrics_dp) + '\n')

    def validate(self):
        # Commented out because we don't use batch norms
        self.model.eval()
        val_loss = 0
        train_acc = 0
        total = 0
        with tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f'Epoch: {self.curr_epoch}/{self.max_epoch} - Valid') as t:
            for batch_idx, (data, target) in t:
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = self.model(data)
                    pred = torch.argmax(output, dim=1)
                    correct = torch.sum(pred == target).item()
                    train_acc += correct
                loss = self.criterion(output, target)
                loss_data = loss.item()
                val_loss += loss_data
                total += len(data)
        return {'val_loss': val_loss / total, 'val_acc': train_acc / total}





def run_experiment(cell_type):
    gpu_device = torch.device('cuda:1')
    path = f'combination_matrix/{cell_type}'
    data_files = sorted(glob(f'{path}/matrix_base*'))
    gene_names = pd.read_csv(f'{path}/geneNames.csv')
    save_dir = f'combination_binary'
    for header in tqdm(data_files):
        matrix_files = [mmread(header).transpose().tocsr().toarray()]
        targets = [0] * matrix_files[-1].shape[0]
        suffix = header.split('matrix_base_')[1]
        
        additional_path = f'{path}/matrix_sim_{suffix}'
        logcounts = mmread(additional_path).transpose().tocsr().toarray()
        matrix_files.append(logcounts)
        targets += ([1] * logcounts.shape[0])
        data = np.concatenate(matrix_files)
        data = stats.zscore(data, axis = 1)
        data[data > 10] = 10
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=18, shuffle=True, stratify=targets)
        train_dataset = MatrixDataset(X_train, y_train)
        valid_dataset = MatrixDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, num_workers=4, batch_size=128, shuffle=True)
        valid_loader = DataLoader(valid_dataset, num_workers=4, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        base_model = Model(data.shape[1], 64, len(matrix_files))
        optim = torch.optim.SGD(base_model.parameters(), lr=0.001, weight_decay=0.01)
        epoch = 100
        save_path = f'{save_dir}/{cell_type}/{suffix}'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        trainer = Trainer(base_model, optim, criterion, train_loader, valid_loader, epoch, gpu_device, save_path)
        trainer.train()
        state = {'state_dict': base_model.state_dict()}
        torch.save(state, f'{save_path}/final.pth')
        
        
cell_types = ["CD4n T", "B"]
from concurrent.futures import ProcessPoolExecutor as Pool
pool = Pool(max_workers=2)
pool.map(run_experiment, cell_types)
# run_experiment("B")

