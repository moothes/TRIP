import argparse
import importlib
import os

def parse_args(model_name):
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    parser.add_argument("model", type=str, default="AttMIL", help="Type of model (Default: mcat)")
    parser.add_argument("--anno_file", type=str, default="/data2/zhouhuajun/tnbc_all/tnbc_13_list.csv", help="Data directory to WSI features (extracted via CLAM)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible experiment (default: 1)")

    # Model Parameters.
    #parser.add_argument("--model_size", type=str, choices=["small", "large"], default="small", help="Size of some models (Transformer)")
    #parser.add_argument("--modal", type=str, default="path,gene,text", help="Specifies which modalities to use / collate function in dataloader.")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="Number of classes")
    parser.add_argument("--task", type=str, choices=["bcls", "cls", "dfs", "os"], default="cls")
    parser.add_argument("--thre", type=float, default=0.5, help="Maximum number of epochs to train (default: 20)")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead", "SGD"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--phase", type=str, choices=["train", "test", "tta"], default="train")
    parser.add_argument("--num_epoch", type=int, default=20, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--loss", type=str, default="nllsurv", help="slide-level classification loss function (default: ce)")
    parser.add_argument("--wsi_norm", type=bool, default=False)
    
    parser.add_argument("--test_set", type=str, choices=["zfy", "tnbc4", "tnbc5", "tnbc6", "tnbc7", "tcga", "cptac", "exsurv"], default="zfy")
    
    
    # Model-specific Parameters
    model_specific_config = importlib.import_module('models.{}.network'.format(model_name)).custom_config
    
    ### Base arguments with customized values
    parser.set_defaults(**model_specific_config['base'])
    
    ### Customized arguments
    for k, v in model_specific_config['customized'].items():
        v['dest'] = k
        parser.add_argument('--' + k, **v)
    
    args = parser.parse_args()
    return args
