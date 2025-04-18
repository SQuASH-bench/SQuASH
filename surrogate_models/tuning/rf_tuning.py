import os
import torch
import optuna
import pickle
import random
from functools import partial

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from util import split
from util.data_loader import load_data
from config import DeviceConfig, get_default_model_config_by_search_space, PathConfig


def prepare_data(data):
    """
    Prepare data by converting inputs to NumPy arrays and flattening features.

    Args:
        data (list or tuple): If a list, assumed to contain tuples (X, y). Otherwise, a tuple
            (X, y) where X can be a tensor.

    Returns:
        tuple: (X, y)
            X is a NumPy array of shape (num_samples, num_features)
            y is a NumPy array of target values.
    """
    if isinstance(data, list):
        X_list, y_list = zip(*data)
        # Convert each X to a NumPy array (if it has a "cpu" method, then assume it's a tensor)
        X = np.stack([x.cpu().numpy() if hasattr(x, "cpu") else x for x in X_list])
        y = np.array(y_list)
    else:
        X, y = data
        if hasattr(X, "cpu"):
            X = X.cpu().numpy()
        if hasattr(y, "cpu"):
            y = y.cpu().numpy()
    # Reshape X to 2D array: (num_samples, num_features)
    X = X.reshape(X.shape[0], -1)
    return X, y


def objective(trial, params, train_data, val_data):
    """
    Objective function for hyperparameter tuning using Optuna.

    This function performs the following:
      - Suggests hyperparameters for the RandomForestRegressor.
      - Trains the model on training data.
      - Evaluates the model on validation data using MSE, R2 and Spearman correlation.
      - Returns the validation metric based on the configuration ('spearman' or 'mse').

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        params (dict): Dictionary of fixed parameters, including seeds, metric to optimize, etc.
        train_data: Training data, formatted as accepted by `prepare_data()`.
        val_data: Validation data, formatted similarly to train_data.

    Returns:
        float: The validation metric value (Spearman correlation or MSE) that Optuna uses for optimization.
    """
    # Suggest hyperparameters to tune
    tuning_params = {
        'n_estimators': trial.suggest_categorical(
            'n_estimators',
            [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 400]
        ),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'n_jobs': -1  # Use all available cores
    }

    print("Tuning parameters:", tuning_params)

    # Instantiate the RandomForestRegressor with the tuned hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=tuning_params['n_estimators'],
        max_depth=tuning_params['max_depth'],
        random_state=params['seed'],
        min_samples_split=tuning_params['min_samples_split'],
        min_samples_leaf=tuning_params['min_samples_leaf'],
        max_features=tuning_params['max_features'],
        n_jobs=tuning_params['n_jobs'],
    )

    # Prepare the training data
    X_train, y_train = prepare_data(train_data)
    rf_model.fit(X_train, y_train)

    # Prepare validation data and make predictions
    X_val, y_val = prepare_data(val_data)
    val_preds = rf_model.predict(X_val)

    # Calculate validation metrics
    val_mse = mean_squared_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)
    val_rho, _ = spearmanr(y_val, val_preds)

    print(f"[Validation] MSE={val_mse:.4f}, R2={val_r2:.4f}, Spearman={val_rho:.4f}")

    # Return the appropriate metric based on configuration
    if params['metric'] == 'spearman':
        return val_rho
    elif params['metric'] == 'mse':
        return val_mse
    else:
        raise ValueError("Invalid metric. Supported metrics: 'spearman' or 'mse'.")


if __name__ == '__main__':
    # Initialize device configuration (if needed by other parts of the code)
    device_config = DeviceConfig()
    device = device_config.device

    # Set the model and gate set identifier based on a pre-defined format (e.g., 'gs1')
    gate_set = 'gs1'
    model_name = f'random_forest_{gate_set}_mse'

    # Load configuration paths and model parameters
    path_config = PathConfig()
    model_config = get_default_model_config_by_search_space(model_name, device)

    # Convert model configuration to a dictionary and add additional paths
    params = vars(model_config)
    params['PATHS'] = path_config.paths

    # Set random seeds for reproducibility
    random.seed(model_config.seed)
    np.random.seed(model_config.seed)
    torch.manual_seed(model_config.seed)

    # Construct the data file path and load data
    data_path = os.path.join(path_config.paths['rf_data'], f'{gate_set}.pt')
    data = load_data(data_path)
    print(f"Data samples: {len(data)}")

    if len(data) == 0:
        raise ValueError("Data is empty.")

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = split.train_val_test_split(
        data, random_seed=model_config.seed
    )
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    # Use partial to bind fixed arguments to the objective function for Optuna
    objective_with_args = partial(objective, params=params, train_data=train_data, val_data=val_data)

    # Determine optimization direction based on chosen metric ('mse' to minimize, 'spearman' to maximize)
    direction = 'minimize' if params['metric'] == 'mse' else 'maximize'
    study_name = f"rf_{params['metric']}_{model_name}"
    storage = f"sqlite:///{os.path.join(path_config.paths['optuna_studies'], study_name)}.db"

    # Create or load an existing Optuna study
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    # Run hyperparameter optimization for 100 trials
    study.optimize(objective_with_args, n_trials=100)

    print("Best hyperparameters: ", study.best_params)
