import os
import sys
import csv
import time
import random
import numpy as np
import importlib
import torch
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')

from utils.options import parse_args
from shutil import copyfile

from torch.utils.data import DataLoader
from utils.loss_factory import Loss_factory
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.engine import Engine
from utils.dataset import TCGA_dataset

from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def main(args):    

    if args.task != 'cls':
        args.loss = 'nllsurv'
        
    dataset = TCGA_dataset(args, 'train')
    train_loader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=False)
                            
    all_results = []
    all_res = []

    if 'cls' in args.task:
        folds = [4, ]
        if args.task == 'bcls':
            args.n_classes = 1
            args.loss = 'BCLSLoss'
    else:
        folds = map(int, args.fold.split(','))
    
    # start 5-fold CV evaluation.
    for fold in folds:
        set_seed(args.seed)
        args.current_fold = fold
        train_loader.dataset.set_train_test(fold)
        torch.cuda.empty_cache()
        
        model = importlib.import_module('models.{}.network'.format(args.model)).Network(args)    
         
        if 'cls' in args.task:
            base_weight_path = 'results/{}_{}/fold_{}/'.format(args.model, args.task, fold)
        else:
            base_weight_path = 'results/{}_{}_{}_/fold_{}/'.format(args.model, args.task, args.test_set, fold)

        file_list = os.listdir(base_weight_path) 
        for file_name in file_list:
            if file_name.endswith('.pth.tar'):
                weight_path = file_name
        print(base_weight_path + weight_path)
        weights = torch.load(base_weight_path + weight_path, weights_only=False)
        
        model.load_state_dict(weights['state_dict'])
        if 'affine_matrix_mean' in weights.keys():
            model.affine_matrix_mean = weights['affine_matrix_mean']
            model.affine_matrix_std = weights['affine_matrix_std']
            model.affine_tta_mean = weights['affine_tta_mean']
            model.affine_tta_std = weights['affine_tta_std']
        
        engine = Engine(args)
        criterion = Loss_factory(args)
        
        # start training
        res, results = engine.deploy(train_loader, model, criterion)
        all_res.append(results)
        if 'cls' in args.task:
            all_results.append([res['ACC'], res['AUC'], res['F1score']])
        else:
            all_results.append([res['cindex']])
            
    
    
    if 'cls' in args.task:
        res_df = pd.DataFrame(all_results, columns=['ACC', 'AUC', 'F1score'])
    else:
        risks = []
        censor = []
        surv_time = []
        for res in all_res:
            for resu in res:
                risks.append(resu[0])
                censor.append(resu[1])
                surv_time.append(resu[2])
        risks = np.array(risks)
        censor = np.array(censor)
        surv_time = np.array(surv_time)
        
        draw_KM(args, risks, 1-censor, surv_time)
        
        res_df = pd.DataFrame(all_results, columns=['cindex'])
    res_df.to_csv('results/{}_{}/{}_{}_test_res.csv'.format(args.model, args.task, args.model, args.test_set))
            
    print(res_df)

def draw_KM(args, hazards, labels, survtime_all):
    # draw KM curve
    from lifelines.statistics import logrank_test
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    # plt.rc('font', family='Times New Roman')
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 40
    plt.rcParams['ytick.labelsize'] = 40
    # from lifelines.plotting import add_at_risk_counts
    from matplotlib.offsetbox import AnchoredText
    # split into high risk group and low risk group
    median = np.median(hazards)
    idx = hazards <= median
    low_risk = survtime_all[idx]
    high_risk = survtime_all[~idx]
    # censorship of each group
    low_label = labels[idx]
    high_label = labels[~idx]
    
    time_interval = [0, 25, 50, 75, 100, 125] 
    for time_int in time_interval:
        actual_live_low = low_label * low_risk + (1 - low_label) * 300
        actual_live_high = high_label * high_risk + (1 - high_label) * 300
        print(sum(actual_live_low > time_int), sum(actual_live_high > time_int))
    # calculate p_valve
    results = logrank_test(low_risk, high_risk, event_observed_A=low_label, event_observed_B=high_label)
    p_value = results.p_value
    # draw curve for each group
    fig, ax = plt.subplots(figsize=(14, 12))
    low = KaplanMeierFitter().fit(low_risk, low_label, label='Low Risk')
    ax = low.plot_survival_function(ax=ax, show_censors=True)
    high = KaplanMeierFitter().fit(high_risk, high_label, label='High Risk')
    ax = high.plot_survival_function(ax=ax, show_censors=True)
    plt.xlabel('Time (Months)', fontsize=40)
    if args.task == 'dfs':
        tt = "Disease-free Survival"
    elif args.task == 'os':
        tt = "Overall Survival"
    plt.ylabel(tt, fontsize=40)
    plt.legend(loc='lower left', fontsize=40)
    plt.ylim(-0.1, 1.1)
    ax.add_artist(AnchoredText("p-value = %.2e" % p_value, loc=4, frameon=False, prop=dict(fontsize=40)))
    plt.tight_layout()
    plt.savefig(os.path.join('./visual/km_curves/{}_{}_{}_KM.pdf'.format(args.model, args.test_set, args.task)), dpi=1000)
    return p_value

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args(sys.argv[1])
    print(args)
    results = main(args)
    print("Finished!")
