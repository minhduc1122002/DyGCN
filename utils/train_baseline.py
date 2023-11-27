import math
import torch
import copy
import numpy as np
from tqdm.auto import tqdm
from utils.loss import compute_loss, compute_roc_auc, compute_ap

def evaluate_step_baseline(model, dataset, spilt, device):
    start, end = dataset.get_range_by_split(spilt)
    test_ap = []
    test_auc = []

    model.eval()
    with torch.no_grad():
        for t in range(start - 1, end - 1):
            current_snapshot = dataset.snapshots[t]
            next_snapshot = dataset.snapshots[t + 1]

            x = current_snapshot.node_feature.to(device)
            edge_index = current_snapshot.edge_index.to(device)

            edge_label_index = next_snapshot.edge_label_index.to(device)
            label = next_snapshot.edge_label.to(device)

            prediction = model(x, edge_index, edge_label_index)

            loss = compute_loss(prediction, label)

            roc_auc = compute_roc_auc(prediction, label)
            ap = compute_ap(prediction, label)

            test_ap.append(ap)
            test_auc.append(roc_auc)

    return np.mean(test_ap), np.mean(test_auc)

def train_step_baseline(model, optimizer, dataset, device):
    start, end = dataset.get_range_by_split('train')
    train_ap = []
    train_auc = []
    count = 0
    train_loss = 0

    model.train()

    for t in range(start, end - 1):
        current_snapshot = dataset.snapshots[t]
        next_snapshot = dataset.snapshots[t + 1]

        x = current_snapshot.node_feature.to(device)
        edge_index = current_snapshot.edge_index.to(device)

        edge_label_index = next_snapshot.edge_label_index.to(device)
        label = next_snapshot.edge_label.to(device)

        prediction = model(x, edge_index, edge_label_index)

        loss = compute_loss(prediction, label)
        roc_auc = compute_roc_auc(prediction, label)
        ap = compute_ap(prediction, label)

        train_ap.append(ap)
        train_auc.append(roc_auc)
        train_loss += loss
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return np.mean(train_ap), np.mean(train_auc)

def train_baseline(model, optimizer, dataset, n_epoch, device):
    best_model = None
    best_model_unchanged = 0
    best_auc = 0
    best_epoch = 0

    for epoch in range(n_epoch):

        train_ap, train_auc = train_step_baseline(model, optimizer, dataset, device)
        val_ap, val_auc = evaluate_step_baseline(model, dataset, 'val', device)
        test_ap, test_auc = evaluate_step_baseline(model, dataset, 'test', device)

        print('Epoch {}: Train: roc_auc: {:.4f}, ap: {:.4f}, Valid: roc_auc: {:.4f}, ap: {:.4f}, Test: roc_auc: {:.4f}, ap: {:.4f}'.format
              (epoch + 1, train_auc, train_ap, val_auc, val_ap, test_auc, test_ap))
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_model_unchanged = 0
        else:
            best_model_unchanged += 1

        if best_model_unchanged >= 20:
            print('Saving Model At Epoch {}'.format(best_epoch + 1))
            break

    model.load_state_dict(best_model)
    final_ap, final_auc = evaluate_step_baseline(model, dataset, 'test', device)

    print('Final Test Results: roc_auc: {:.4f}, ap: {:.4f}'.format(final_auc, final_ap))
    return final_auc, final_ap