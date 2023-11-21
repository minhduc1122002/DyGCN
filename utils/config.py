import argparse
import sys
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='uci',
                        choices=['uci', 'bc-otc', 'bc-alpha', 'enron', 'dblp'])
    
    parser.add_argument('--dataset_interval', help='time interval to spilt data', default='W',
                        choices=['W', 'D', 'M'])
    
    parser.add_argument('--train_ratio', type=float, default=0.7, help='ratio of validation set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of test set')
    parser.add_argument('--input_dim', type=int, default=64, help='randomly generate initial embedding')

    parser.add_argument('--model_name', type=str, default='SubTree', help='name of the model',
                        choices=['SubTree', 'EvolveGCN', 'Roland'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--window_size', type=int, default=5, help='merge window in subtree model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--egcn_rnn', type=str, default='GRU', help='EvolveGCN RNN type')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args