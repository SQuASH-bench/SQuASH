import json
import os
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from config import get_default_model_config_by_search_space, PathConfig
from surrogate_models.architectures.gnn.gnn_model import RegGNN
from util import split
from util.config_utils import get_gate_set_and_features_by_name
from util.data_loader import load_data, save_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_initial_best_valid_metrics(metric):
    if metric == 'spearman':
        return -float('inf')
    if metric in {'loss', 'r2'}:
        return float('inf')
    raise ValueError("Invalid metric. Choose from 'spearman', 'loss', or 'r2'.")


def load_best_model(model, path, device):
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def convert_json_serializable(obj):
    """
    Recursively convert non-serializable values (e.g., torch.device) to strings.
    """
    if isinstance(obj, dict):
        return {k: convert_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_json_serializable(v) for v in obj]
    elif isinstance(obj, torch.device):
        return str(obj)
    else:
        return obj


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = convert_json_serializable(obj)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def train(model_name, params, train_loader, val_loader, test_loader, loss_fn):
    print(f"Device: {params['device']}")
    model = RegGNN(
        num_layer=params['layer_num'],
        emb_dim=params['emb_dim'],
        edge_attr_dim=params['qubit_num'],
        num_node_features=params['num_node_features'],
        drop_ratio=params['drop_ratio'],
        graph_pooling=params['graph_pooling'],
        JK=params['JK'],
        freeze_gnn=False
    ).to(params['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
    best_metric = set_initial_best_valid_metrics(params['metric'])
    best_model_path = os.path.join(params['PATHS']['trained_models'], f'{model_name}.pth')

    records = {k: [] for k in ['train_losses', 'val_losses', 'train_spearmans', 'val_spearmans', 'train_r2', 'val_r2']}
    patience, overfit, n_imp = params['patience'], 0, 0
    prev_train_loss, prev_val_loss = None, None

    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch + 1}/{params['epochs']}")

        train_loss, train_spear, train_r2 = RegGNN.train_one_epoch(model, train_loader, optimizer, loss_fn,
                                                                   params['device'])
        val_loss, val_spear, val_r2 = RegGNN.evaluate(model, val_loader, loss_fn, params['device'], desc="Validation")

        for k, v in zip(records.keys(), [train_loss, val_loss, train_spear, val_spear, train_r2, val_r2]):
            records[k].append(v)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Train Spearman: {train_spear:.4f}, Validation Spearman: {val_spear:.4f}")

        if (
                (params['metric'] == 'spearman' and val_spear > best_metric) or
                (params['metric'] == 'loss' and val_loss < best_metric) or
                (params['metric'] == 'r2' and val_r2 < best_metric)
        ):
            best_metric = {'spearman': val_spear, 'loss': val_loss, 'r2': val_r2}[params['metric']]
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to {best_model_path}")
            n_imp = 0
        else:
            n_imp += 1

        if prev_train_loss is not None and train_loss < prev_train_loss and val_loss >= prev_val_loss:
            overfit += 1
            print(f"[INFO] Overfitting detected ({overfit}/{patience - 2})")
        else:
            overfit = 0

        if overfit >= patience - 2 or n_imp >= patience:
            print("[INFO] Early stopping")
            break

        prev_train_loss, prev_val_loss = train_loss, val_loss

    model = load_best_model(model, best_model_path, params['device'])
    test_loss, test_spear, test_r2 = RegGNN.evaluate(model, test_loader, loss_fn, params['device'], desc="Testing")

    print(f"\nTest Loss: {test_loss:.4f}, Spearman: {test_spear:.4f}, R2: {test_r2:.4f}")
    results = [(model_name, best_metric, test_loss, test_spear, test_r2)]
    return results, {model_name: records}


def plot_metrics(epochs, records, key, ylabel, title):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, records[f'train_{key}'], label=f'Train {ylabel}')
    plt.plot(epochs, records[f'val_{key}'], label=f'Validation {ylabel}')
    if key == 'losses':
        plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    search_space = 'ghz_a'
    model_name = f"gcn_{search_space}_{timestamp}"
    gate_set, features = get_gate_set_and_features_by_name(f"gate_set_{search_space}")
    print(f"[INFO] Training on gate set: {gate_set}")

    paths = PathConfig().paths
    model_config = get_default_model_config_by_search_space(model_type="gcn",
                                                            search_space=search_space,
                                                            features=features,
                                                            device=device)
    config = vars(model_config)
    config['PATHS'], config['device'], config['timestamp'] = paths, device, timestamp

    set_seed(model_config.runseed)

    data_name = f"test_dataset_{search_space}"
    data_path = os.path.join(paths['gcn_data'], f'{data_name}.pt')
    print(f"Loading data from {data_path}")
    data = load_data(data_path)

    if not data:
        raise ValueError("[ERROR] Data is empty")

    train_data, val_data, test_data = split.train_val_test_split(data, random_seed=model_config.seed)
    print(f"[INFO] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    save_data(os.path.join(paths['gcn_data'], f'test_gcn_{search_space}_{timestamp}.pt'), test_data)

    train_loader = DataLoader(train_data, batch_size=model_config.batch_size, shuffle=True,
                              num_workers=model_config.num_workers)
    val_loader = DataLoader(val_data, batch_size=model_config.batch_size, shuffle=False,
                            num_workers=model_config.num_workers)
    test_loader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=False,
                             num_workers=model_config.num_workers)

    save_json(os.path.join(paths['trained_models'], f'{model_name}_config.json'), config)

    print("[INFO] Starting training")
    results, loss_records = train(model_name, config, train_loader, val_loader, test_loader, nn.MSELoss())

    for model_name, best_val, test_loss, test_spear, test_r2 in results:
        print(
            f"\nModel: {model_name} | Best Val: {best_val:.4f} | Test Loss: {test_loss:.4f} | Spearman: {test_spear:.4f} | R2: {test_r2:.4f}")

    benchmark_path = os.path.join(paths['benchmark'], f'gcn_{search_space}_benchmark_results.pkl')
    records_path = os.path.join(paths['benchmark'], f'gcn_{search_space}_records.pkl')
    with open(benchmark_path, 'wb') as f:
        pickle.dump(results, f)
    with open(records_path, 'wb') as f:
        pickle.dump(loss_records, f)
    print("loss_records", loss_records)

    epochs = range(1, len(loss_records[model_name]['train_losses']) + 1)
    plot_metrics(epochs, loss_records[model_name], 'losses', 'Loss', 'Training and Validation Loss')
    plot_metrics(epochs, loss_records[model_name], 'spearmans', "Spearman's rho", "Training and Validation Spearman's rho")
