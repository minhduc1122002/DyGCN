## Overview

This is the implementation of DyGSTA from the paper "Temporal Structural Preserving with Subtree Attention in Dynamic Graph
Transformers"

## Requirements
```
pip install -r requirements.txt
```
## Training

* Example of training DyGSTA on UCI dataset:
```
python main.py --dataset_name uci --model_name DyGSTA --window_size 5 --num_runs 3
```

* Example of training [Roland](https://arxiv.org/pdf/2208.07239.pdf) on UCI dataset:
```
python main.py --dataset_name uci --model_name ROLAND --num_runs 3
```
