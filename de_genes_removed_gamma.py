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
from matplotlib_venn import venn3, venn3_unweighted
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

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
    
class Model2(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(Model2, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_output)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.do1 = torch.nn.Dropout(0.5)

    def forward(self, x):
        return self.linear3(self.act2(self.linear2(self.do1(self.act1(self.linear1(x))))))
    

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
        best_val_acc = -float('inf')
        for epoch in range(1, self.max_epoch+1):
            self.curr_epoch = epoch
            metrics = self.train_epoch()
            if metrics['val_acc'] > best_val_acc:
                best_val_acc = metrics['val_acc']
        return best_val_acc

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

import copy

def get_acc(data, targets, remove, gpu_device, save_path):
    new_data = copy.deepcopy(data)
    if remove:
        new_data[:, np.array(remove)] = 0
    X_train, X_test, y_train, y_test = train_test_split(new_data, targets, test_size=0.2, random_state=18, shuffle=True, stratify=targets)

    #clf = LogisticRegression() # Lasso
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    lr_val_acc = accuracy_score(y_test, y_pred)
    
    train_dataset = MatrixDataset(X_train, y_train)
    valid_dataset = MatrixDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, num_workers=4, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    base_model = Model(data.shape[1], 64, 2)
#     base_model = Model2(data.shape[1], 1000, 500, 2)
#     optim = torch.optim.SGD(base_model.parameters(), lr=0.001, weight_decay=0.05, momentum=0.1, nesterov=True)
    optim = torch.optim.Adam(base_model.parameters(), lr=0.001, weight_decay=0.01)

    epoch = 100

    trainer = Trainer(base_model, optim, criterion, train_loader, valid_loader, epoch, gpu_device, save_path, 1)
    nn_val_acc = trainer.train()
#     import ipdb; ipdb.set_trace()
    return lr_val_acc, nn_val_acc
    
def run_experiment(cell_type):
    gpu_device = torch.device('cuda:0')
    path = f'combination_matrix_gamma/{cell_type}'
    data_files = sorted(glob(f'{path}/matrix_base*'))
    
    save_dir = f'combination_binary_gamma'
    de_dir = f'de_analysis_combination_binary_gamma'
    nn_result = []
    lr_result = []
    gene_names = pd.read_csv(f'{path}/geneNames.csv')['gene_names'].tolist()
    gene_to_idx = {v:k for k,v in enumerate(gene_names)}
    col_names = ['F', 'E+F', 'D+E', 'B+E', 'A+B+D+E', 'B+C+E+F', 'A+B+C+D+E+F', 'A+B+C+D+E+F+G']
    
    for header in tqdm(data_files):
        if "NumCells_1000_NumDE_100_min_lfc_1.2_max_lfc_3" not in header:
            continue
        gene_size = []
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
        
        save_path = f'{save_dir}/{cell_type}/{suffix}'
        saliency_df = pd.read_csv(f'{save_path}/new_saliency_group_1.csv')
        save_path_de = f'{de_dir}/{cell_type}/{suffix}'
        de_method_df = pd.read_csv(f'{save_path_de}/rankings_group_2.csv')
        
        de_genes = "{}_de_genes.csv".format(suffix.split('.txt')[0])
        gene_set = {ind-1 for ind in pd.read_csv(f'{path}/{de_genes}')['gene_index']}
        saliency_list = saliency_df.loc[saliency_df['Vanilla'] <= len(gene_set), "gene_names"].tolist()
        saliency_genes = {gene_to_idx[g] for g in saliency_list}
        de_method_list = de_method_df['limma'].iloc[:len(gene_set)].tolist()
        de_method_genes = {gene_to_idx[g] for g in de_method_list}
        
        E = all_sets = gene_set & saliency_genes & de_method_genes # E
        D = de_saliency = (gene_set & saliency_genes) - all_sets # D
        F = de_limma = (gene_set & de_method_genes) - all_sets # F
        B = saliency_limma = (saliency_genes & de_method_genes) - all_sets # B
        A = saliency_only = saliency_genes - all_sets - de_saliency - saliency_limma # A
        C = limma_only = de_method_genes - all_sets - saliency_limma - de_limma # C
        G = de_only = gene_set - all_sets - de_limma - de_saliency # G
        

        plt.figure(figsize=(10,10))
        # onlyset1, onlyset2, set1+set2, onlyset3, set1+set3, set2+set3, all
        subset = tuple(map(lambda x: len(x), [saliency_only, limma_only, saliency_limma, de_only, de_saliency, de_limma, all_sets]))
        plot = venn3_unweighted(subsets = subset, set_labels = ('Saliency', 'Limma', 'DE'))
        plot.get_label_by_id('100').set_text(f'A ({len(A)})')
        plot.get_label_by_id('010').set_text(f'C ({len(C)})')
        plot.get_label_by_id('110').set_text(f'B ({len(B)})')
        plot.get_label_by_id('001').set_text(f'G ({len(G)})')
        plot.get_label_by_id('101').set_text(f'D ({len(D)})')
        plot.get_label_by_id('011').set_text(f'F ({len(F)})')
        plot.get_label_by_id('111').set_text(f'E ({len(E)})')
        plt.savefig(f'{save_path}/venn.png')
        #continue
        
        total = set(range(logcounts.shape[1]))
        
        F_acc = get_acc(data, targets, list(total-F), gpu_device, save_path)
        gene_size.append(len(F))
        
        EF_acc = get_acc(data, targets, list(total-F-E), gpu_device, save_path)
        gene_size.append(len(F|E))
        
        DE_acc = get_acc(data, targets, list(total-D-E), gpu_device, save_path)
        gene_size.append(len(D|E))
        
        BE_acc = get_acc(data, targets, list(total-B-E), gpu_device, save_path)
        gene_size.append(len(B|E))
        
        ABDE_acc = get_acc(data, targets, list(total-A-B-D-E), gpu_device, save_path)
        gene_size.append(len(A|B|D|E))
        
        BCEF_acc = get_acc(data, targets, list(total-B-C-E-F), gpu_device, save_path)
        gene_size.append(len(F|E|B|C))
        
        ABCDEF_acc = get_acc(data, targets, list(total-F-E-A-B-D-C), gpu_device, save_path)
        gene_size.append(len(F|E|A|B|C|D))
        
        ABCDEFG_acc = get_acc(data, targets, list(total-F-E-A-B-D-C-G), gpu_device, save_path)
        gene_size.append(len(F|E|A|B|C|D|G))

        
        lr_accuracies = []
        nn_accuracies = []
        for acc in [F_acc, EF_acc, DE_acc, BE_acc, ABDE_acc, BCEF_acc, ABCDEF_acc, ABCDEFG_acc]:
            lr_acc, nn_acc = acc
            lr_accuracies.append(lr_acc)
            nn_accuracies.append(nn_acc)
            
        pd.DataFrame({
            'svm_acc': lr_accuracies, 
            'nn_accuracies': nn_accuracies,
            'experiment': col_names,
            'gene_count': gene_size
            
        }).to_csv(f'{save_path}/comparisons.csv')

                    
        
# cell_types = ["CD14 Monocyte", "NK", "CD4n T", "B", "DC", "CD8m T", "CD4m T", "CD16 Monocyte", "pDC", "CD8eff T", "Platelet", "Neutrophil", "IgG PB", "IgA PB", "Activated Granulocyte", "SC & Eosinophil", "RBC"]
# from concurrent.futures import ProcessPoolExecutor as Pool
# pool = Pool(max_workers=8)
# pool.map(run_experiment, cell_types)

run_experiment("B")
