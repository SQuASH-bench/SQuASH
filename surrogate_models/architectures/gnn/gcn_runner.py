import os
import torch
import pickle
import random
import matplotlib.pyplot as plt

import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader

from util import split
from surrogate_models.architectures.gnn.gnn_model import RegGNN
from util.data_loader import load_data, save_data
from config import get_model_config, PathConfig
from datetime import datetime


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def set_initial_best_valid_metrics(params):
    if params['metric'] == 'spearman':
        best_val_metric = -float('inf')
    elif params['metric'] == 'loss':
        best_val_metric = float('inf')
    elif params['metric'] == 'r2':
        best_val_metric = float('inf')
    else:
        raise ValueError("Invalid metric. Supported metrics: 'spearman' or 'loss' or 'r2'")
    return best_val_metric


def load_best_model(model, best_model_path, params):
    model.load_state_dict(torch.load(best_model_path))
    model.to(params['device'])
    return model


def fit(model_name, params, train_loader, val_loader, test_loader, loss_fn):
    """
    Train and evaluate the RegGNN model.

    Args:
        model_name (str): Identifier for the model (used in saving the model file).
        params (dict): Dictionary of hyperparameters and configuration settings.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        loss_fn (function): Loss function (e.g., nn.MSELoss) used for training.

    Returns:
        tuple: A tuple containing:
            - results (list): A list of tuples with model results (model name, best validation metric,
                              test loss, test Spearman’s rho, and test R2 score).
            - loss_records (dict): A dictionary recording training and validation loss, Spearman’s rho,
                                   and R2 scores for each epoch.
    """
    print(f"Device: {params['device']}")

    reg_model = RegGNN(
        num_layer=params['layer_num'],
        emb_dim=params['emb_dim'],
        edge_attr_dim=params['qubit_num'],
        num_node_features=params['num_node_features'],
        drop_ratio=params['drop_ratio'],
        graph_pooling=params['graph_pooling'],
        JK=params['JK'],
        freeze_gnn=False
    )

    reg_model.to(params['device'])

    optimizer = torch.optim.Adam(reg_model.parameters(), lr=params['lr'], weight_decay=params['decay'])

    # initialize lists to record loss and metric values
    train_losses, val_losses, train_spearmans, val_spearmans, train_r2s, val_r2s = [], [], [], [], [], []

    # early stopping configuration
    patience = params['patience']
    overfit = 0
    n_imp = 0  # number of epochs without improvement

    best_val_metric = set_initial_best_valid_metrics(params=params)
    best_model_path = os.path.join(params['PATHS']['trained_models'], f'gcn_{model_name}_{timestamp}.pth')

    prev_train_loss, prev_val_loss = None, None

    # Training loop over epochs.
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch + 1}/{params['epochs']}")

        # train for one epoch.
        train_loss, train_spearman, train_r2 = RegGNN.train_one_epoch(
            reg_model, train_loader, optimizer, loss_fn, params['device']
        )
        # evaluate on the valid set
        val_loss, val_spearman, val_r2 = RegGNN.evaluate(
            reg_model, val_loader, loss_fn, params['device'], desc="Validation"
        )

        # Record the losses and metrics.
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_spearmans.append(train_spearman)
        val_spearmans.append(val_spearman)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Train Spearman's rho: {train_spearman:.4f}, Validation Spearman's rho: {val_spearman:.4f}")

        # check and update best model based on the chosen metric
        if params['metric'] == 'spearman':
            if val_spearman > best_val_metric:
                best_val_metric = val_spearman
                torch.save(reg_model.state_dict(), best_model_path)
                print("[INFO] Saving best model")
                n_imp = 0
            else:
                n_imp += 1
        elif params['metric'] == 'loss':
            if val_loss < best_val_metric:
                best_val_metric = val_loss
                torch.save(reg_model.state_dict(), best_model_path)
                print("[INFO] Saving best model")
                n_imp = 0
            else:
                n_imp += 1
        elif params['metric'] == 'r2':
            if val_r2 < best_val_metric:
                best_val_metric = val_r2
                torch.save(reg_model.state_dict(), best_model_path)
                print("[INFO] Saving best model")
                n_imp = 0
            else:
                n_imp += 1

        # check for overfitting: if training loss decreases but validation loss doesn't improve
        if prev_train_loss is not None and prev_val_loss is not None:
            if train_loss < prev_train_loss and val_loss >= prev_val_loss:
                overfit += 1
                print(f"[INFO] Overfitting detected ({overfit}/{patience - 2}).")
            else:
                overfit = 0

        prev_train_loss, prev_val_loss = train_loss, val_loss

        # early stopping
        if overfit >= patience - 2 or n_imp >= patience:
            print("Early stopping triggered")
            break

    # load the best model for testing
    reg_model = load_best_model(model=reg_model, best_model_path=best_model_path, params=params)

    # evaluate the model on the test set
    test_loss, test_spearman, test_r2 = RegGNN.evaluate(reg_model, test_loader, loss_fn, params['device'],
                                                        desc="Testing")

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Spearman's rho: {test_spearman:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    # prepare final results and loss records
    results = [('gcn', best_val_metric, test_loss, test_spearman, test_r2)]
    loss_records = {
        'gcn': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_spearmans': train_spearmans,
            'val_spearmans': val_spearmans,
            'train_r2': train_r2s,  # Note: Recorded per epoch
            'val_r2': val_r2s,  # Note: Recorded per epoch
        }
    }

    return results, loss_records


if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    gate_set = 'ghz_a'
    gate_set_name = "test_dataset_ghz_a"  # Dataset name based on gate set.
    model_name = f'{gate_set_name}'
    model_conf = f'gcn_{gate_set}'
    print(f"[INFO] Training on gate set: {gate_set_name}")

    path_config = PathConfig()
    model_config = get_model_config(model_conf, device)
    config_dict = vars(model_config)
    config_dict['PATHS'] = path_config.paths

    # set random seeds for reproducibility
    random.seed(model_config.seed)
    torch.manual_seed(model_config.runseed)
    np.random.seed(model_config.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_config.runseed)

    # load dataset from the specified directory
    data_dir = os.path.join(path_config.paths['gcn_data'], f'{gate_set_name}.pt')
    print(f"Load data from {data_dir}")
    data = load_data(data_dir)
    print(f"Total data samples: {len(data)}")

    if len(data) == 0:
        raise ValueError("Data is empty.")

    # split the dataset
    train_data, val_data, test_data = split.train_val_test_split(data, random_seed=model_config.seed)
    print(
        f"[INFO] Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    # save test data for further reference.
    save_data(os.path.join(path_config.paths['gcn_data'], f'test_gcn_{gate_set_name}_{timestamp}.pt'), test_data)
    print(f"[INFO] Test Data saved to: {os.path.join(path_config.paths['test_data'])}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=model_config.batch_size, shuffle=True,
                              num_workers=model_config.num_workers)
    val_loader = DataLoader(val_data, batch_size=model_config.batch_size, shuffle=False,
                            num_workers=model_config.num_workers)
    test_loader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=False,
                             num_workers=model_config.num_workers)

    # define the loss function
    criterion = nn.MSELoss()

    print(f"[INFO] Start training with: {model_name}")
    print(
        f"[INFO] Using params:\n"
        f"  Embedding dim: {model_config.emb_dim}\n"
        f"  Number of layers: {model_config.layer_num}\n"
        f"  Edge attr dim (qubit num): {model_config.qubit_num}\n"
        f"  Num node features: {model_config.num_node_features}\n"
        f"  Learning rate: {model_config.lr}\n"
        f"  Batch size: {model_config.batch_size}\n"
        f"  Dropout ratio: {model_config.drop_ratio}\n"
        f"  Graph pooling: {model_config.graph_pooling}\n"
        f"  JK: {model_config.JK}"
    )

    # fit the model and obtain results
    results, loss_records = fit(model_name, config_dict, train_loader, val_loader, test_loader, criterion)

    # print final performance results
    for model_name, best_val_metric, test_loss, test_spearman, test_r2 in results:
        print(f"\nModel: {model_name} --- "
              f"Best Validation Metric: {best_val_metric:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Spearman's rho: {test_spearman:.4f}, "
              f"Test R2: {test_r2:.4f}")

    # save benchmark results and training records for future analysis
    benchmark_results_path = os.path.join(path_config.paths['benchmark'], f'gcn_{gate_set}_benchmark_results.pkl')
    records_path = os.path.join(path_config.paths['benchmark'], f'gcn_{gate_set}_records.pkl')
    with open(benchmark_results_path, 'wb') as f:
        pickle.dump(results, f)
    with open(records_path, 'wb') as f:
        pickle.dump(loss_records, f)

    # plot training and validation losses
    epochs = range(1, len(loss_records['gcn']['train_losses']) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss_records['gcn']['train_losses'], label='Train Loss')
    plt.plot(epochs, loss_records['gcn']['val_losses'], label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot training and validation Spearman's rho
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss_records['gcn']['train_spearmans'], label="Train Spearman's rho")
    plt.plot(epochs, loss_records['gcn']['val_spearmans'], label="Validation Spearman's rho")
    plt.xlabel('Epoch')
    plt.ylabel("Spearman's rho")
    plt.title("Training and Validation Spearman's rho")
    plt.legend()
    plt.grid(True)
    plt.show()
