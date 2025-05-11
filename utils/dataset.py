from __future__ import print_function, division
import os
import math
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer

eps = 1e-5

subtype_tcga_list = ['LUMA', 'LUMB', 'NA', 'HER2', 'TNBC']
subtype_list = ['LUMA', 'LUMB1', 'LUMB2', 'HER2', 'TNBC']

def is_nan(s):
    try:
        num = float(s)
        return math.isnan(num)
    except ValueError:
        return False

class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    metadata: Optional[dict] = None

class TCGA_dataset(Dataset):
    def __init__(self, args='', phase='train'):
        self.anno_file = args.anno_file
                
        self.phase = phase
        self.args = args
        self.task = args.task

        self.data_list = pd.read_csv(self.anno_file)
        print('Totally {} samples loaded.'.format(len(self.data_list)))
        if 'cls' not in args.task:
            self.data_list = self.data_list[self.data_list['subtype']=='TNBC']
            self.data_list.dropna(subset=[args.task + '_time', args.task + '_status'], inplace=True)
            self.get_label(args.task, args.n_classes)
            
        self.get_path_mean()
        self.set_train_test(fold=0)
        
        self.fake_status = torch.rand((len(self.data_list))) > 0.9

    def get_label(self, target='os', ncls=10):
        tar_column = target + '_time'
        uncensored_df = self.data_list[self.data_list[tar_column] > 0]

        disc_labels, q_bins = pd.qcut(uncensored_df[tar_column], q=ncls, retbins=True, labels=False)
        q_bins[-1] = self.data_list[tar_column].max() + eps
        q_bins[0] = self.data_list[tar_column].min() - eps

        disc_labels, q_bins = pd.cut(self.data_list[tar_column], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.data_list[target + '_label'] = disc_labels.values.astype(int)
        
    def set_train_test(self, fold=0):
        self.fold = fold
        if self.task == 'bcls':
            tag = 'cls'
        else:
            tag = self.task
        if self.args.test_set == 'zfy':
            if self.args.phase == 'train':
                self.train_list = self.data_list[self.data_list[tag + '_split'] != self.fold]
                self.test_list = self.data_list[self.data_list[tag + '_split'] == self.fold]
            else:
                self.train_list = self.data_list[self.data_list[tag + '_split'] == self.fold]
                self.test_list = self.data_list[self.data_list[tag + '_split'] == self.fold]
        else:
            self.train_list = self.data_list
            self.test_list = self.data_list

    def __iter__(self):
        self.test_idx = 0
        return self
    
    def __next__(self):
        cur_idx = self.test_idx
        self.test_idx += 1
        return self.get_sample(cur_idx)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index):
        if self.phase in ['train', 'tta']:
            data_row = self.train_list.iloc[index]
        else:
            data_row = self.test_list.iloc[index]

        pid = data_row['patient_id']
        if self.task == 'cls':
            if self.args.test_set == 'tcga':
                label = subtype_tcga_list.index(data_row['subtype'].upper())
            else:
                label = subtype_list.index(data_row['subtype'].upper())
        elif self.task == 'bcls':
            label = int(data_row['subtype'].upper() == 'TNBC')
        else:
            label = data_row[self.task + '_label']
            
        path_files = data_row['path'].split(';')
        path_feat = []
        for pfile in path_files:
            if 'resnet50' in pfile:
                pfile = pfile.replace("resnet50", "gpfm")
            path_feat.append(torch.load(pfile, weights_only=False))
        path_feat = torch.concat(path_feat, dim=0)
        
        if self.args.model == 'tnbcmil':
            if len(path_feat) > 30000:
                selected = random.sample(range(len(path_feat)), 30000)
                path_feat = path_feat[selected]
                    
        if self.args.wsi_norm:
            path_feat = (path_feat - self.path_mean) / (self.path_std + 1e-10)
        
        if 'cls' not in self.task:
            event_time = float(data_row[self.task + '_time'])
            status = 1 - torch.tensor(float(data_row[self.task + '_status'])) # Code is based on MCAT, where censorship is reversed with status
        else:
            event_time, status = -1, -1

        return path_feat.float(), label, event_time, status, pid

    def get_gene_embed(self, gene_list):
        pass

    def get_path_mean(self):
        self.path_mean = torch.load('/data2/zhouhuajun/tnbc_all/tnbc_gpfm_mean.pt', weights_only=True).view(1, -1)
        self.path_std = torch.load('/data2/zhouhuajun/tnbc_all/tnbc_gpfm_std.pt', weights_only=True).view(1, -1)

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)
    