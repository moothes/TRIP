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
        seed_list = range(1000, 10000, 1000)
        #print(seed_list)
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

    results_dir = "./results/{model}_{task}_tta_{dataset}".format(model=args.model, task=args.task, dataset=args.test_set)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.results_dir = results_dir
    csv_path = os.path.join(results_dir, "results_all.csv")
    args.phase = 'tta'
    args.loss = 'pseloss'
    args.num_epoch = 20
    args.n_classes = 1
    
    if args.task == 'cls':
        summary_df = pd.DataFrame(columns=['dataset', 'fold', 'ACC', 'AUC', 'F1score'])
    else:
        summary_df = pd.DataFrame(columns=['dataset', 'fold', 'cindex'])
        
    #for fold in folds:
    for fold in [4, ]:
        set_seed(args.seed)
        dataset.set_train_test(fold)
        
        model = importlib.import_module('models.{}.network'.format(args.model)).Network(args)
        engine = Engine(args)
        criterion = Loss_factory(args)
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)
        
        base_weight_path = 'results/{}_{}_new/fold_{}/'.format(args.model, args.task, fold)
        file_list = os.listdir(base_weight_path)
        for file_name in file_list:
            if file_name.endswith('.pth.tar'):
                weight_path = file_name
        model.load_state_dict(torch.load(base_weight_path + weight_path, weights_only=False)['state_dict'], strict=False)
        print('Load weight from {}.'.format(base_weight_path + weight_path))
        
        result = engine.tta(model, dataset, criterion, optimizer, scheduler, fold, 'tta')
        result['dataset'] = args.test_set
        result['fold'] = fold
        
        res_dict = pd.DataFrame({'name':result.keys(), 'value':result.values()})
        summary_df = pd.concat([summary_df, res_dict], ignore_index=True)
        
        print('Overall: {}.'.format(result))

    summary_df.to_csv(csv_path, index=False)
    return csv_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args(sys.argv[1])
    results = run_for_seed(args)
    print("Finished!")
