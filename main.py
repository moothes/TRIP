import os
import sys
import csv
import time
import random
import torch
import numpy as np
import pandas as pd
import importlib

from shutil import copyfile

from utils.options import parse_args
from utils.loss_factory import Loss_factory
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.engine import Engine
from utils.dataset import TCGA_dataset

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

def run_for_seed(args):
    print_res = {}
    if args.seed == -1:
        seed_list = range(50, 10000, 100)
        for s in seed_list:
            args.seed = s
            print_res[args.seed] = main(args)
    else:
        print_res[args.seed] = main(args)
    print(print_res)

def main(args):
    print(args)
    folds = list(map(int, args.fold.split(',')))
    dataset = TCGA_dataset(args)


    if args.phase == 'tta':
        results_dir = "./results/{model}_{task}_{tset}_{lr}_{time}".format(model=args.model, task=args.task, tset=args.test_set, lr=args.lr, time=time.strftime("%Y-%m-%d]-[%H-%M-%S"))
    else:
        results_dir = "./results/{model}_{task}_{lr}_{time}".format(model=args.model, task=args.task, lr=args.lr, time=time.strftime("%Y-%m-%d]-[%H-%M-%S"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.results_dir = results_dir
    csv_path = os.path.join(results_dir, "results_all.csv")
    
    summary_df = []
    counter = 0
    if args.task == 'cls':
        args.loss = 'CLSLoss'
    elif args.task == 'bcls':
        args.n_classes = 1
        args.loss = 'BCLSLoss'
        args.num_epoch = 5
    else:
        args.loss = 'nllsurv'
        
    if args.phase == 'tta':
        args.num_epoch = 20
    
    for fold in folds:
        set_seed(args.seed)
        dataset.set_train_test(fold)
        
        model = importlib.import_module('models.{}.network'.format(args.model)).Network(args)
        
        if args.phase == 'tta':
            if args.task == 'os':
                base_weight_path = 'results/{}_{}_/fold_{}/'.format(args.model, args.task, fold)
            elif args.task == 'dfs':
                base_weight_path = 'results/{}_{}_0.0002_2025-04-05]-[01-00-40/fold_{}/'.format(args.model, args.task, fold)
            else:
                base_weight_path = 'results/{}_{}/fold_{}/'.format(args.model, args.task, fold)
                
            file_list = os.listdir(base_weight_path)
            for file_name in file_list:
                if file_name.endswith('.pth.tar'):
                    weight_path = file_name
            weights = torch.load(base_weight_path + weight_path, weights_only=False)
            model.load_state_dict(weights['state_dict'])
            print('TTA loading weights from {}.'.format(base_weight_path + weight_path))
        
        
        engine = Engine(args)
        criterion = Loss_factory(args)
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)
        
        result = engine.learning(model, dataset, criterion, optimizer, scheduler, fold, phase=args.phase)
        result['dataset'] = args.test_set
        result['fold'] = fold
        res_dict = pd.DataFrame([result])
        summary_df.append(res_dict)
        
        print('Overall: {}.'.format(result))

    sdf = pd.concat(summary_df)
    sdf.to_csv(csv_path, index=False)
    
    return csv_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args(sys.argv[1])
    results = run_for_seed(args)
    print("Finished!")
