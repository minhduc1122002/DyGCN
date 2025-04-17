## Overview

A Simple baseline for dynamic graph link prediction

## Requirements
```
pip install -r requirements.txt
```
## Training

* Example of training DyGCN on UCI dataset:
```
python main.py --dataset_name uci --model_name DyGCN --window_size 5 --num_runs 3 --n_layers 2 --hidden_dim 64
```
