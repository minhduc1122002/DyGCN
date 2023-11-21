import math
import torch
import copy
import numpy as np
from utils.loss import compute_loss, compute_roc_auc, compute_ap

def train_step(model, optimizer, dataset, device):
    start, end = dataset.get_range_by_split('train')
    train_loss = 0
    count = 0
    train_ap = []
    train_auc = []

    model.train()
    model.reset_memory()

    init_state = torch.zeros(dataset.snapshots[0].node_feature.size())
    
    for t in range(start, end - 1):
        current_snapshot = dataset.snapshots[t]
        next_snapshot = dataset.snapshots[t + 1]

        x = current_snapshot.node_feature.to(device)
        init_state = init_state.to(device)
        edge_index = current_snapshot.edge_index.to(device)
        edge_feature = current_snapshot.edge_time.to(device)

        edge_label_index = next_snapshot.edge_label_index.to(device)
        label = next_snapshot.edge_label.to(device)

        prediction, new_state = model(x, edge_index, edge_label_index, edge_feature, init_state)

        loss = compute_loss(prediction, label)

        init_state = new_state.detach().cpu().clone()

        roc_auc = compute_roc_auc(prediction, label)
        ap = compute_ap(prediction, label)

        train_ap.append(ap)
        train_auc.append(roc_auc)
        train_loss += loss
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_ap), np.mean(train_auc), init_state, model.memory_edge_index, model.memory_edge_time

def evaluate_step(model, dataset, previous_state, previous_memory_edge_index, previous_memory_edge_time, spilt, device):
    start, end = dataset.get_range_by_split(spilt)
    test_ap = []
    test_auc = []

    init_state = previous_state.clone()

    model.read_memory(copy.deepcopy(previous_memory_edge_index), copy.deepcopy(previous_memory_edge_time))
    model.eval()
    with torch.no_grad():
        for t in range(start - 1, end - 1):
            current_snapshot = dataset.snapshots[t]
            next_snapshot = dataset.snapshots[t + 1]

            x = current_snapshot.node_feature.to(device)
            init_state = init_state.to(device)
            edge_index = current_snapshot.edge_index.to(device)
            edge_feature = current_snapshot.edge_time.to(device)
            
            edge_label_index = next_snapshot.edge_label_index.to(device)
            label = next_snapshot.edge_label.to(device)

            prediction, new_state = model(x, edge_index, edge_label_index, edge_feature, init_state)

            init_state = new_state.detach().cpu().clone()

            roc_auc = compute_roc_auc(prediction, label)
            ap = compute_ap(prediction, label)

            test_ap.append(ap)
            test_auc.append(roc_auc)

    if spilt == 'val':
        return np.mean(test_ap), np.mean(test_auc), init_state, model.memory_edge_index, model.memory_edge_time
    else:
        return np.mean(test_ap), np.mean(test_auc)

def train(model, optimizer, dataset, n_epoch, device):
    best_model = None
    best_model_unchanged = 0
    best_auc = 0
    best_epoch = 0
    best_state = None

    for epoch in range(n_epoch):
        train_ap, train_auc, train_state, train_memory_edge_index, train_memory_edge_time = train_step(model, optimizer, dataset, device)
        val_ap, val_auc, val_state, val_memory_edge_index, val_memory_edge_time = evaluate_step(model, dataset, train_state, train_memory_edge_index, train_memory_edge_time, 'val', device)
        test_ap, test_auc = evaluate_step(model, dataset, val_state, val_memory_edge_index, val_memory_edge_time, 'test', device)

        print('Epoch {}: Train: roc_auc: {:.4f}, ap: {:.4f}, Valid: roc_auc: {:.4f}, ap: {:.4f}, Test: roc_auc: {:.4f}, ap: {:.4f}'.format
              (epoch + 1, train_auc, train_ap, val_auc, val_ap, test_auc, test_ap))
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_state = val_state
            best_model_unchanged = 0
        else:
            best_model_unchanged += 1

        if best_model_unchanged >= 20:
            print('Saving Model At Epoch {}'.format(best_epoch + 1))
            break

    model.load_state_dict(best_model)
    
    final_ap, final_auc = evaluate_step(model, dataset, best_state, val_memory_edge_index, val_memory_edge_time, 'test', device)

    print('Final Test Results: roc_auc: {:.4f}, ap: {:.4f}'.format(final_auc, final_ap))
    return final_auc, final_ap