import time
import torch
import copy
import numpy as np
from tqdm import tqdm

from utils.loss import compute_loss, compute_roc_auc, compute_ap
from dataloader.utils import negative_sampling

def train_step(model, optimizer, dataset, device):
    start, end = dataset.get_range_by_split('train')
    train_loss = 0
    count = 0
    train_ap = []
    train_auc = []

    model.train()
    model.reset_memory()

    init_state = torch.zeros(dataset.snapshots[0].node_feature.shape[0], model.hidden_dim)

    for t in tqdm(range(start, end - 1), leave=False):
        current_snapshot = dataset.snapshots[t]
        next_snapshot = dataset.snapshots[t + 1]

        x = current_snapshot.node_feature.to(device)
        init_state = init_state.to(device)
        edge_index = current_snapshot.edge_index.to(device)
        edge_feature = current_snapshot.edge_time.to(device)

        negative_edge_index = negative_sampling(next_snapshot.edge_index)

        edge_label_index = torch.cat([next_snapshot.edge_index, negative_edge_index], dim=-1).to(device)
        label = torch.cat([torch.ones(next_snapshot.edge_index.shape[1]),
                           torch.zeros(negative_edge_index.shape[1])]).to(device)

        # edge_label_index = next_snapshot.edge_label_index.to(device)
        # label = next_snapshot.edge_label.to(device)

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

    train_loss = train_loss / count

    train_metric = {
        'loss': train_loss,
        'ap': np.mean(train_ap),
        'auc': np.mean(train_auc),
    }
    return train_metric, init_state, model.memory_edge_index, model.memory_edge_time

def evaluate_step(model, dataset, previous_state, previous_memory_edge_index, previous_memory_edge_time, spilt, device):
    start, end = dataset.get_range_by_split(spilt)
    test_loss = 0
    count = 0
    test_ap = []
    test_auc = []

    init_state = previous_state.clone()

    model.read_memory(copy.deepcopy(previous_memory_edge_index), copy.deepcopy(previous_memory_edge_time))

    model.eval()
    with torch.no_grad():
        for t in tqdm(range(start - 1, end - 1), leave=False):
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
            test_loss += loss
            count += 1

            test_ap.append(ap)
            test_auc.append(roc_auc)

    test_loss = test_loss / count

    test_metric = {
        'loss': test_loss,
        'ap': np.mean(test_ap),
        'auc': np.mean(test_auc),
    }

    if spilt == 'val':
        return test_metric, init_state, model.memory_edge_index, model.memory_edge_time
    else:
        return test_metric

def train(model, optimizer, dataset, n_epoch, patience, device):
    best_model = None
    best_model_unchanged = 0
    best_auc = 0
    best_epoch = 0
    best_state = None

    train_auc = []
    valid_auc = []
    for epoch in range(n_epoch):

        start_time = time.time()

        train_metric, train_state, train_memory_edge_index, train_memory_edge_time = train_step(model, optimizer, dataset, device)

        val_metric, val_state, val_memory_edge_index, val_memory_edge_time = evaluate_step(model, dataset, train_state, train_memory_edge_index,
                                                                                           train_memory_edge_time, 'val', device)

        test_metric = evaluate_step(model, dataset, val_state, val_memory_edge_index, val_memory_edge_time, 'test', device)

        epoch_time = time.time() - start_time

        print('Epoch {}, Time: {}s'.format(epoch + 1, epoch_time))

        print('Train: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(train_metric['loss'], train_metric['auc'], train_metric['ap']))

        print('Valid: loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(val_metric['loss'], val_metric['auc'], val_metric['ap']))

        print('Test : loss: {:.4f}, roc_auc: {:.4f}, ap: {:.4f}'.format(test_metric['loss'], test_metric['auc'], test_metric['ap']))

        print('======' * 20)

        train_auc.append(train_metric['auc'])
        valid_auc.append(val_metric['auc'])

        if val_metric['auc'] > best_auc:
            best_auc = val_metric['auc']
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_state = val_state
            best_model_unchanged = 0
        else:
            best_model_unchanged += 1

        if best_model_unchanged >= patience:
            print('Saving Model At Epoch {}'.format(best_epoch + 1))
            break

    model.load_state_dict(best_model)

    final_metric = evaluate_step(model, dataset, best_state, val_memory_edge_index, val_memory_edge_time, 'test', device)

    print('Final Test Results: roc_auc: {:.4f}, ap: {:.4f}'.format(final_metric['auc'], final_metric['ap']))

    return final_metric