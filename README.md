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

The data is spilt with the ratio of 70/10/20 for training, validation, testing. All methods have the same training settings and is tested on 4 datasets over 3 seeds:

ROC-AUC:

| Method             |      UCI     |    BC-OTC    |   BC-ALPHA   |   MATH-OF    |
|--------------------|--------------|--------------|--------------|--------------|
| EvolveGCN-H        | 66.97 ± 1.46 |              |              |              |
| EvolveGCN-O        | 72.20 ± 3.18 |              |              |              |
| Roland             | 87.93 ± 1.29 | 94.77 ± 0.39 | 94.50 ± 0.14 | 81.91 ± 2.17 |
| SubTree (window=1) | 92.18 ± 0.42 | 95.71 ± 0.45 | 95.20 ± 0.72 | 93.34 ± 0.03 |
| SubTree (window=5) | 93.41 ± 0.49 | 96.90 ± 0.42 | 96.78 ± 0.15 | 95.27 ± 0.34 |
| SubTree (window=10)| 95.09 ± 0.04 | 96.05 ± 0.72 | 96.38 ± 0.22 |              |

AP:

| Method             |      UCI     |    BC-OTC    |   BC-ALPHA   |   MATH-OF    |
|--------------------|--------------|--------------|--------------|--------------|
| EvolveGCN-H        | 73.08 ± 0.65 |              |              |              |
| EvolveGCN-O        | 75.94 ± 1.55 |              |              |              |
| Roland             | 90.26 ± 0.86 | 95.76 ± 0.32 | 95.29 ± 0.19 | 86.67 ± 1.44 |
| SubTree (window=1) | 93.16 ± 0.16 | 96.80 ± 0.21 | 96.05 ± 0.60 | 95.18 ± 0.03 |
| SubTree (window=5) | 93.81 ± 0.30 | 97.64 ± 0.24 | 97.12 ± 0.11 | 96.07 ± 0.27 |
| SubTree (window=10)| 94.90 ± 0.15 | 97.09 ± 0.36 | 96.96 ± 0.15 |              |

## TODO
- [ ] Embedding-based Sampling/Merging
- [ ] Embedding edge time to SubTree Attention
- [ ] More experiments on different benchmarks
- [ ] More baselines (GCRN, DySAT, etc)