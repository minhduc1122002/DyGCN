## Overview

SubTree Aggregation + Gru + Merge Graph

## Requirements
```
pip install -r requirements.txt
```
## Training

* Example of training SubTree on UCI dataset:
```
python main.py --dataset_name uci --model_name SubTree --window_size 5 --num_runs 3
```

* Example of training [Roland](https://arxiv.org/pdf/2208.07239.pdf) on UCI dataset:
```
python main.py --dataset_name uci --model_name Roland --num_runs 3
```

* Example of training [EvolveGCN](https://arxiv.org/pdf/1902.10191.pdf) on UCI dataset:
```
python main.py --dataset_name uci --model_name EvolveGCN --egcn_rnn GRU --num_runs 3
```

## Results
  
### Link Prediction

The data is spilt with the ratio of 70/10/20 for training, validation, testing. All methods have the same training settings, hidden dim, etc.
The results are averaged over 3 seeds:

| Method   | UCI (ROC-AUC) | UCI (AP) |
|----------|---------------|----------|
| EvolveGCN-H | 66.97 ± 1.46 | 73.08 ± 0.65 |
| EvolveGCN-O | 72.20 ± 3.18 | 75.94 ± 1.55 |
| Roland | 88.26 ± 1.58 | 89.63 ± 1.26 |
| SubTree (window=1, no attention) | 92.00 ± 0.11 | 93.33 ± 0.18 |
| SubTree (window=5, no attention) | 94.20 ± 0.93 | 94.39 ± 0.60 |

## TODO
- [ ] Embedding-based Sampling/Merging
- [ ] Embedding edge time to SubTree Attention
- [ ] More experiments on different benchmarks
- [ ] More baselines (GCRN, DySAT, etc)