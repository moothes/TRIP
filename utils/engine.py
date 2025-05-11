import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
import time
import json
import pandas as pd
import math

from sklearn.cluster import KMeans

from utils.loss_factory import metric

import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

torch.set_num_threads(4)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Engine(object):
    def __init__(self, args):
        self.args = args
        self.best_score = 0
        self.best_epoch = 0
        self.best_res = 0
        self.filename_best = None

    def learning(self, model, dataset, criterion, optimizer, scheduler, fold, phase='train'):
        fold_dir = os.path.join(self.args.results_dir, 'fold_' + str(fold))
        if not os.path.isdir(fold_dir):
            os.mkdir(fold_dir)

        if torch.cuda.is_available():
            model = model.cuda()
        torch.cuda.empty_cache()

        self.dy_thre = self.args.thre
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train one epoch
            self.run_epoch(dataset, model, criterion, phase=phase, optimizer=optimizer)
            # test one epoch
            result = self.run_epoch(dataset, model, criterion, phase='eval')
            if 'cls' in self.args.task:
                main_metric = 'AUC'
            else:
                main_metric = 'cindex'
                
            score = result[main_metric]
            
            if score >= self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self.best_res = result
                    
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'affine_matrix_mean': model.affine_matrix_mean,
                    'affine_matrix_std': model.affine_matrix_std,
                    'affine_tta_mean': model.affine_tta_mean,
                    'affine_tta_std': model.affine_tta_std,
                    'best_score': score,
                    'score': result,
                    'fold': fold,})
            print(' *** Current c-index={:.4f}, best c-index={:.4f} at epoch {}'.format(score, self.best_score, self.best_epoch))
            print('>')
        return self.best_res

    def run_epoch(self, dataset, model, criterion, phase='train', optimizer=None, tta_thre=0.5):
        if phase == 'eval':
            eval('model.eval()')
        else:
            eval('model.train()')
            
        dataset.phase = phase
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=phase in ['train', 'tta'], num_workers=4, pin_memory=True, drop_last=False)
        num_samples = len(data_loader)
        
        sum_loss = 0.0
        all_loss_dict = {}
        for k in criterion.loss_collection.keys():
            all_loss_dict[k] = 0
            
        results = []
        progressbar = tqdm(range(num_samples), desc='{} {} samples for epoch {}'.format(phase, num_samples, self.epoch), ncols=150)
        
        batch_cnt = 0
        print(self.dy_thre)
        for index, (data_WSI, label, event_time, status, pid) in zip(progressbar, data_loader):
            wsis = data_WSI.cuda()
            label = label.cuda()
            ss = label
            
            if phase in ['train', 'tta']:
                out = model(x_path=wsis, phase=phase)
            else:
                with torch.no_grad():
                    out = model(x_path=wsis, phase=phase)
            
            if self.args.task == 'cls':
                out['cls'] = torch.softmax(out['pred'], dim=-1)
                _, pred_cls = torch.max(out['cls'], dim=-1)
                results.append([out['cls'].detach().cpu().numpy(), pred_cls.detach().cpu().numpy(), label.detach().cpu().numpy()])
            if self.args.task == 'bcls':
                out['cls'] = torch.sigmoid(out['pred'])
                if phase == 'tta':
                    out['pse'] = (out['cls'] > self.dy_thre).float()
                label = label.unsqueeze(-1)
                pred_cls = (out['cls'] > self.args.thre).long()
                results.append([out['cls'].detach().cpu().numpy(), pred_cls.detach().cpu().numpy(), label.detach().cpu().numpy()])
            else:
                out['hazards'] = torch.sigmoid(out['pred'])
                out['S'] = torch.cumprod(1 - out['hazards'], dim=-1)
                new_label = torch.argmax(out['pred']).view(1)
                if phase == 'tta':
                    label = new_label
                
                risk = -torch.sum(out['S'], dim=1)
                results.append([risk.detach().cpu().numpy(), status, event_time])
            loss, loss_dict = criterion(out, {'label': label, 'c': status.cuda()})
            batch_cnt += 1

            pats = []
            for k, v in loss_dict.items():
                all_loss_dict[k] += v
                pats.append('{}: {:.4f}'.format(k, all_loss_dict[k] / (index + 1)))
            str_loss = ', '.join(pats)
            sum_loss += loss.item()
            
            if phase in ['train', 'tta'] and batch_cnt == 1:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_str = optimizer.param_groups[-1]['lr']
                progressbar.set_postfix_str('LR: {:.1e}, {}'.format(lr_str, str_loss))
                batch_cnt = 0
                torch.cuda.empty_cache()
            
        # calculate loss and error for epoch
        sum_loss /= len(progressbar)
        res = metric(results, self.args.task, testset=self.args.test_set)
        print('loss: {:.4f}, {}'.format(sum_loss, res))
        return res

    def deploy(self, data_loader, model, criterion):
        model = model.cuda()
        model.eval()
        val_loss = 0.0
        results = []
        features = []
        logits = []
        labels = []
        hazards = []
        censor = []
        serv_time = []
        confs = []
        pclses = []
        visual_dict = {'all_knowledge': [], 'all_grad': [], 'all_id': []}
        progressbar = tqdm(data_loader, desc='Validating ', ncols=150)
        
        pids = []
        risk_dict = []
        events = []
        statuses = []
        
        for batch_idx, (data_WSI, label, event_time, status, pid) in zip(progressbar, data_loader):
            label = label.float().cuda()
            data_WSI = data_WSI.float().cuda()

            with torch.no_grad():
                    out = model(x_path=data_WSI, phase='test')
                    
            if self.args.task == 'cls':
                out['cls'] = torch.softmax(out['pred'], dim=-1)
                
                _, pred_cls = torch.max(out['cls'], dim=-1)
                results.append([out['cls'].detach().cpu().numpy(), pred_cls.detach().cpu().numpy(), label.detach().cpu().numpy()])
            if self.args.task == 'bcls':
                out['cls'] = torch.sigmoid(out['pred'])
                label = label.unsqueeze(-1)
                pred_cls = (out['cls'] > self.args.thre).long()
                results.append([out['cls'].detach().cpu().numpy(), pred_cls.detach().cpu().numpy(), label.detach().cpu().numpy()])
                
                pids.append(pid[0])
                confs.append(out['cls'].detach().cpu().numpy()[0][0])
                pclses.append(pred_cls.detach().cpu().numpy()[0][0])
                labels.append(label.detach().cpu().numpy()[0][0])
            else:
                out['hazards'] = torch.sigmoid(out['pred'])
                out['S'] = torch.cumprod(1 - out['hazards'], dim=-1)
                
                risk = -torch.sum(out['S'], dim=1).detach().cpu().numpy()
                results.append([risk, status.numpy(), torch.ceil(event_time).numpy()])
                
                risk_dict.append(risk[0])
                pids.append(pid[0])
                statuses.append(status.numpy()[0])
                events.append(event_time.numpy()[0])
        
        val_loss /= len(progressbar)
        res = metric(results, self.args.task, testset=self.args.test_set)
        print('loss: {:.4f}, result: {}'.format(val_loss, res))
        return res, results
        
    def tta(self, model, dataset, criterion, optimizer, scheduler, fold, phase='tta'):
        fold_dir = os.path.join(self.args.results_dir, 'fold_' + str(fold))
        if not os.path.isdir(fold_dir):
            os.mkdir(fold_dir)

        if torch.cuda.is_available():
            model = model.cuda()
        torch.cuda.empty_cache()

        thres = {'tnbc4': 0.01, 'tnbc6': 0.1, 'tcga': 0.95}
        dyn_thre = thres[self.args.test_set]
                
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train one epoch
            confs = []
            _, res = self.run_epoch(dataset, model, criterion, phase=phase, optimizer=optimizer, tta_thre=dyn_thre)
            for re in res:
                confs.append(re[0][0])
            confs = np.array(confs)
            dyn_thre = np.percentile(confs, 70) # np.mean(confs) #
            print('New threshold: {}.'.format(dyn_thre))
            # test one epoch
            result = self.run_epoch(dataset, model, criterion, phase='eval', tta_thre=dyn_thre)
            if 'cls' in self.args.task:
                main_metric = 'ACC'
            else:
                main_metric = 'cindex'
                
            score = result[main_metric]
            
            if score >= self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self.best_res = result
                    
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': score,
                    'score': result,
                    'fold': fold,})
            print(' *** Current c-index={:.4f}, best c-index={:.4f} at epoch {}'.format(score, self.best_score, self.best_epoch))
            print('>')
        return self.best_res
        
    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.args.results_dir, 'fold_' + str(state['fold']), 'epoch{}_{:.4f}.pth.tar'.format(state['epoch'], state['best_score']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)

def collate_custom(batch):

    path_feat = [item[0].float() for item in batch] 
    gene_feat = [item[1].float() for item in batch] 

    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    pid = np.array([item[5] for item in batch])

    return path_feat, gene_feat, label, event_time, c, pid