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
import copy
from captum._utils.models.linear_model import SkLearnLinearRegression


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
        #self.act = nn.LeakyReLU()
        self.act = nn.ReLU()

    def forward(self, x):
        h_relu = self.act(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred
    
def run_saliency(cell_type):
    
    gpu_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = f'combination_matrix_gamma/{cell_type}'
    data_files = sorted(glob(f'{path}/matrix_base*'))
    gene_names = pd.read_csv(f'{path}/geneNames.csv')
    save_dir = f'combination_binary_gamma'
    for header in tqdm(data_files):
        if "repeat" not in header:
            continue
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
        base_model = Model(data.shape[1], 64, len(matrix_files))
        save_path = f'{save_dir}/{cell_type}/{suffix}'

        save = torch.load(f'{save_path}/final.pth', map_location=gpu_device)
        
        base_model.load_state_dict(save['state_dict'])
        base_model.eval()
        base_model.to(gpu_device)

        scores = defaultdict(list)
        prediction = []
        methods = ['GradientShap', 'Saliency', 'Deconvolution', 'GuidedBackprop', 'InputXGradient', 'DeepLift', 'IntegratedGradients']
        

        complete_dataset = MatrixDataset(data, targets)
        complete_loader = DataLoader(complete_dataset, num_workers=4, batch_size=32, shuffle=False)
        for data, target in tqdm(complete_loader):
            with torch.no_grad():
                data, target = data.to(gpu_device), target.to(gpu_device)
                output = base_model(data)
                pred = torch.argmax(output, dim=1).tolist()
                prediction += pred
            data.requires_grad_()
            for method in methods:
#                 import ipdb; ipdb.set_trace()
                if method == 'Vanilla':
                    pass
                else:
                    sm_object = getattr(attr, method)(base_model)
#                     if method in {'Lime'}:
                        
                    if method in {'DeepLiftShap', 'GradientShap'}:
                        base_lines = torch.randn(10, data.shape[1])
                        base_lines = base_lines.to(gpu_device)
                        saliency = sm_object.attribute(data, base_lines, target=pred)
                    elif method in {'Lime'}:
                        sm_object = getattr(attr, method)(base_model, interpretable_model = SkLearnLinearRegression())
                        saliency_list = []
                        for i in range(data.shape[0]):
                            saliency_list.append(sm_object.attribute(data[i].reshape(1,-1), target=pred[i]))
                        saliency = torch.cat(saliency_list)
                    elif method == 'Occlusion':
                        saliency = sm_object.attribute(data, target=pred, sliding_window_shapes = (1,))
                    else:
                        saliency = sm_object.attribute(data, target=pred)
                rd = rankdata(saliency.cpu().detach().abs(), axis=1, method = 'ordinal')
                rd = rd.shape[1] - rd
                scores[method].append(rd)

        pd.DataFrame({'predictions': prediction}).to_csv(f'{save_path}/predictions.csv')
        prediction_np = np.array(prediction)

        #for i in range(1,2): # needs to change
        for i in range(1):
            bi = prediction_np == i
            gene_names_copy = gene_names.copy()
            for method in methods:
                mean_rank = np.concatenate(scores[method])[bi].mean(axis=0)
                gene_names_copy[method] = rankdata(mean_rank, axis= 0, method =  'ordinal')
            gene_names_copy.to_csv(f'{save_path}/new_saliency_group_{i}_all.csv')
        
# cell_types = ["CD14 Monocyte", "NK", "CD4n T", "B", "DC", "CD8m T", "CD4m T", "CD16 Monocyte", "pDC", "CD8eff T", "Platelet", "Neutrophil", "IgG PB", "IgA PB", "Activated Granulocyte", "SC & Eosinophil", "RBC"]
# cell_types = ["B", "CD4n T"]
# from concurrent.futures import ProcessPoolExecutor as Pool
# pool = Pool(max_workers=2)
# pool.map(run_saliency, cell_types)
run_saliency("B")
run_saliency("CD4n T")