import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import numpy as np
import warnings

from model.models import ROLAND, EvolveGCN, Model
from utils.train_baseline import train_baseline
from utils.train import train
from utils.train_roland import train_roland
from utils.utlis import fix_seed, count_parameters, get_dataset
from utils.config import get_args

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

        if args.model_name == 'Roland':
            model = ROLAND(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers)

        elif args.model_name == 'EvolveGCN':
            model = EvolveGCN(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers,
                              num_nodes=dataset.num_nodes, rnn=args.egcn_rnn)
            
        elif args.model_name == 'SubTree':
            model = Model(dim_in=args.input_dim, hidden_dim=args.hidden_dim, dim_out=1, num_layer=args.num_layers,
                          window_size=args.window_size)

        model.to(args.device)

        if run == 0:
          print(model)

        print('======' * 20)
        print('Total Parameters: {}'.format(count_parameters(model)))
        print('======' * 20)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.model_name == 'Roland':
            avg_auc, avg_ap = train_roland(model, optimizer, dataset, args.num_epochs, args.device)
        elif args.model_name == 'EvolveGCN':
            avg_auc, avg_ap = train_baseline(model, optimizer, dataset, args.num_epochs, args.device)
        else:
            avg_auc, avg_ap = train(model, optimizer, dataset, args.num_epochs, args.device)

        auc_run.append(avg_auc * 100)
        ap_run.append(avg_ap * 100)

    print('======' * 20)
    print('Final Results: roc_auc: {:.4f} ± {:.4f}, ap: {:.4f} ± {:.4f}'.format(np.mean(auc_run), np.std(auc_run),
                                                                                np.mean(ap_run), np.std(ap_run)))