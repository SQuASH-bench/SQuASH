import os
import torch
import optuna
import random
from functools import partial

import torch.nn as nn
import numpy as np

from torch_geometric.loader import DataLoader

from surrogate_models.architectures.gnn.gnn_model import RegGNN
from util import split
from util.data_loader import load_data
from config import get_default_model_config_by_search_space, PathConfig


def objective(trial, params, train_loader, val_loader, criterion):
    """
    Objective function for Optuna hyperparameter optimization.

    This function defines the training and validation loop for a single trial,
    tuning the hyperparameters of a RegGNN model. It reports the validation
    metric to Optuna and uses early stopping based on a patience parameter.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object.
        params (dict): A dictionary containing model parameters and other config values.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (function): Loss function to optimize (e.g., MSELoss).

    Returns:
        float: The best validation metric (either maximized Spearman correlation or minimized MSE).

    Raises:
        optuna.exceptions.TrialPruned: If the trial is pruned by Optuna.
    """
    # Define hyperparameters to tune
    tuning_params = {
        'emb_dim': trial.suggest_categorical(
            'emb_dim', [800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400]
        ),
        'layer_num': trial.suggest_int('num_layers', 4, 12),
        'drop_ratio': trial.suggest_float('drop_ratio', 0.0, 0.5),
        'graph_pooling': trial.suggest_categorical('graph_pooling', ['sum', 'mean', 'max', 'attention']),
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
        'decay': trial.suggest_float('decay', 1e-4, 1e-1),
        'JK': trial.suggest_categorical('JK', ['mean', 'last'])
    }

    # Instantiate the RegGNN model with the tuned hyperparameters
    reg_model = RegGNN(
        num_layer=tuning_params['layer_num'],
        emb_dim=tuning_params['emb_dim'],
        edge_attr_dim=params['qubit_num'],
        num_node_features=params['num_node_features'],
        drop_ratio=tuning_params['drop_ratio'],
        graph_pooling=tuning_params['graph_pooling'],
        JK=tuning_params['JK'],
        freeze_gnn=False
    )
    reg_model.to(params['device'])

    # Initialize the optimizer with learning rate and weight decay from tuning params
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=tuning_params['lr'], weight_decay=tuning_params['decay'])

    # Set up early stopping parameters
    patience = params.get('patience', 7)
    epochs_without_improvement = 0

    # Initialize best validation metric depending on the metric type
    if params['metric'] == 'spearman':
        best_val_metric = -float('inf')
    elif params['metric'] == 'mse':
        best_val_metric = float('inf')
    else:
        raise ValueError("Invalid metric. Supported metrics: 'spearman' or 'mse'")

    print("Tuning parameters:", tuning_params)

    # Training loop for defined number of epochs
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch + 1}/{params['epochs']}")
        # Train for one epoch
        train_loss, train_spearman, train_r2 = RegGNN.train_one_epoch(
            reg_model, train_loader, optimizer, criterion, params['device']
        )
        # Evaluate on validation data
        val_loss, val_spearman, val_r2 = RegGNN.evaluate(
            reg_model, val_loader, criterion, params['device'], desc="Validation"
        )

        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        if params['metric'] == 'spearman':
            print(f"Train Spearman: {train_spearman:.4f}, Val Spearman: {val_spearman:.4f}")

        # Report the validation metric to Optuna for pruning decisions
        if params['metric'] == 'spearman':
            trial.report(val_spearman, step=epoch)
        elif params['metric'] == 'mse':
            trial.report(val_loss, step=epoch)

        # Prune trial if not promising
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        # Check if validation performance improved
        improved = False
        if params['metric'] == 'spearman':
            if val_spearman > best_val_metric:
                best_val_metric = val_spearman
                improved = True
        else:  # 'mse'
            if val_loss < best_val_metric:
                best_val_metric = val_loss
                improved = True

        # Update early stopping counter based on improvement
        if not improved:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        # Trigger early stopping if no improvement within patience
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return best_val_metric


if __name__ == '__main__':
    # Set dataset and gate set identifier
    gate_set = 'gs1'
    dataset = 'gs1'

    # Device configuration: use GPU device if available, otherwise CPU.
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Load configuration paths and model parameters
    path_config = PathConfig()
    model_config = get_default_model_config_by_search_space(f'gcn_{gate_set}', device)

    # Convert model_config to a dictionary; also add paths from configuration.
    params = vars(model_config)
    params['PATHS'] = path_config.paths

    # Set random seeds for reproducibility
    random.seed(model_config.seed)
    torch.manual_seed(model_config.runseed)
    np.random.seed(model_config.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_config.runseed)

    # Load dataset using the provided data loader
    data = load_data(os.path.join(path_config.paths['gcn_data'], f'{dataset}.pt'))
    print(f"Data samples: {len(data)}")

    if len(data) == 0:
        raise ValueError("Data is empty.")

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = split.train_val_test_split(data, random_seed=model_config.seed)
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    # Create DataLoaders for each set
    train_loader = DataLoader(
        train_data, batch_size=model_config.batch_size, shuffle=True,
        num_workers=model_config.num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=model_config.batch_size, shuffle=False,
        num_workers=model_config.num_workers
    )
    test_loader = DataLoader(
        test_data, batch_size=model_config.batch_size, shuffle=False,
        num_workers=model_config.num_workers
    )

    # Define the loss function for regression
    criterion = nn.MSELoss()

    # Set up the Optuna pruner using median pruner strategy
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

    # Bind additional arguments to the objective function using partial.
    objective_with_args = partial(objective, params=params, train_loader=train_loader, val_loader=val_loader,
                                  criterion=criterion)

    # Define study direction based on the optimization metric
    direction = 'minimize' if params['metric'] == 'mse' else 'maximize'
    study_name = f"gcn_{params['metric']}_" + dataset
    storage = f"sqlite:///{os.path.join(path_config.paths['optuna_studies'], study_name)}.db"

    # Create or load an existing Optuna study
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )

    # Run optimization across a specified number of trials
    study.optimize(objective_with_args, n_trials=100)

    print("Best hyperparameters: ", study.best_params)
