import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def compute_loss(logits, labels):
    fn = torch.nn.BCEWithLogitsLoss()
    logits = logits.flatten()
    labels = labels.flatten().float()
    return fn(logits, labels)

def compute_roc_auc(logits, labels):
    probabilities = torch.sigmoid(logits)
    return roc_auc_score(y_true=labels.detach().cpu().numpy(), y_score=probabilities.detach().cpu().numpy())

def compute_ap(logits, labels):
    probabilities = torch.sigmoid(logits)
    return average_precision_score(y_true=labels.detach().cpu().numpy(), y_score=probabilities.detach().cpu().numpy())

def compute_accuracy(logits, labels):
    pred_label = torch.zeros(len(logits))
    logits = logits.flatten()
    pred_label[logits >= 0.5] = 1.0
    accuracy = np.mean(labels.detach().cpu().numpy() == pred_label.numpy())
    return accuracy