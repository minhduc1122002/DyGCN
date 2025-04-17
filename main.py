import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import numpy as np
import warnings

from model.DyGCN import DyGCN
from model.DySAT import DySAT
from model.GCRN import GCRN
from model.ROLAND import ROLAND
from model.EvolveGCN import EvolveGCN
from model.TGCN import TGCN
from model.WinGNN import WinGNN

from utils.train_baseline import train_baseline
from utils.train import train
from utils.train_roland import train_roland
from dataloader.utils import fix_seed, count_parameters
from utils.config import get_args, get_dataset

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # get arguments
    args = get_args()

    auc_run = []
    ap_run = []

    for run in range(args.num_runs):
        seed = run + 1
        fix_seed(seed)
        print('Run {}, Seed {}'.format(run + 1, run + 1))
        print('======' * 20)

        torch.cuda.empty_cache()

        dataset = get_dataset(args)

        if args.model_name == 'ROLAND':
            model = ROLAND(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers)

        elif args.model_name == 'EvolveGCN':
            model = EvolveGCN(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers,
                              num_nodes=dataset.num_nodes, rnn=args.rnn)
        
        elif args.model_name == 'GCRN':
            model = GCRN(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers,
                        num_nodes=dataset.num_nodes, rnn=args.rnn)
        
        elif args.model_name == 'DySAT':
            model = DySAT(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers,
                          time_step=dataset.num_snapshots)

        elif args.model_name == 'TGCN':
            model = TGCN(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers)
            
        elif args.model_name == 'WinGNN':
            model = WinGNN(dim_in=args.input_dim, hidden_dim=args.hidden_dim, num_layer=args.num_layers)
            
        elif args.model_name == 'DyGCN':
            model = DyGCN(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_heads=args.num_heads, n_layers=args.n_layers,
                          window_size=args.window_size, recurrent=args.recurrent, time_encode=args.time_encode, device=args.device)

        model.to(args.device)

        if run == 0:
          print(model)

        print('======' * 20)
        print('Total Parameters: {}'.format(count_parameters(model)))
        print('======' * 20)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.model_name == 'ROLAND':
            avg_auc, avg_ap = train_roland(model, args, optimizer, dataset, args.num_epochs, args.device)
        elif args.model_name in ['EvolveGCN', 'GCRN', 'DySAT', 'TGCN']:
            avg_auc, avg_ap = train_baseline(model, args, optimizer, dataset, args.num_epochs, args.device)
        else:
            avg_auc, avg_ap = train(model, optimizer, dataset, args.num_epochs, args.patience, args.device)

        auc_run.append(avg_auc * 100)
        ap_run.append(avg_ap * 100)

    print('======' * 20)
    print('Final Results: roc_auc: {:.2f} ± {:.2f}, ap: {:.2f} ± {:.2f}'.format(
        np.mean(auc_run), np.std(auc_run),
        np.mean(ap_run), np.std(ap_run),
    ))