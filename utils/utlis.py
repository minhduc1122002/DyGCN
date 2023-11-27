import random
import torch
import numpy as np

from dataloader.dataset import UCIDataset, BitCoinDataset, DBLPDataset, MathOFDataset

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataset(args):
    
    if args.dataset_name == 'uci':
        URL = 'http://snap.stanford.edu/data/CollegeMsg.txt.gz'
        FILE_NAME = 'CollegeMsg.txt.gz'
        dataset = UCIDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)

    elif args.dataset_name == 'bc-otc':
        URL = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
        FILE_NAME = 'soc-sign-bitcoinotc.csv.gz'
        dataset = BitCoinDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)
        
    elif args.dataset_name == 'bc-alpha':
        URL = 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz'
        FILE_NAME = 'soc-sign-bitcoinotc.csv.gz'
        dataset = BitCoinDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)

    elif args.dataset_name == 'dblp':
        URL = 'https://snap.stanford.edu/data/sx-mathoverflow.txt.gz'
        FILE_NAME = 'sx-mathoverflow.txt.gz'
        dataset = MathOFDataset(URL, FILE_NAME, args.dataset_interval, args.train_ratio, args.val_ratio)
    else:
        raise NotImplementedError('No such dataset: {}'.format(args.dataset_name))
    
    return dataset