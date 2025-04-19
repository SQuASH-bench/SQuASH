# Copyright 2025 Fraunhofer Institute for Open Communication Systems FOKUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import random
import torch

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from config import DeviceConfig, PathConfig, get_default_model_config_by_search_space
from util import split
from util.data_loader import load_data, save_data


def prepare_data(data):
    """
    Prepare data for training/evaluation by converting it to NumPy arrays.

    Args:
        data (list or tuple): Input data; either a list of (X, y) tuples or a tuple (X, y).

    Returns:
        tuple: (X, y) where X is a 2D NumPy array and y is a NumPy array.
    """
    if isinstance(data, list):
        X_list, y_list = zip(*data)
        X = np.stack([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in X_list])
        y = np.array(y_list)
    else:
        X, y = data
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

    X = X.reshape(X.shape[0], -1)
    return X, y


def train_model(params, train_data):
    """
    Train a RandomForestRegressor model using the provided training data and parameters.

    Args:
        params (dict): Dictionary of hyperparameters and configuration settings.
        train_data (list or tuple): Training data to be used for model fitting.

    Returns:
        RandomForestRegressor: The trained RandomForestRegressor model.
    """
    X_train, y_train = prepare_data(train_data)
    rf_model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=params['seed'],
        min_samples_split=params.get('min_samples_split', 2),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        max_features=params.get('max_features', "auto"),
        n_jobs=params.get('n_jobs', -1)
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(model, data, desc="Evaluation"):
    """
    Evaluate a trained model on given data and compute regression metrics.

    Args:
        model (RandomForestRegressor): A trained RandomForestRegressor model.
        data (list or tuple): Evaluation data.
        desc (str, optional): Description for the evaluation (default "Evaluation").

    Returns:
        tuple: (mse, rho, r2) containing the computed Mean Squared Error,
               Spearman's correlation, and R2 score.
    """
    X_eval, y_eval = prepare_data(data)
    preds = model.predict(X_eval)

    mse = mean_squared_error(y_eval, preds)
    r2 = r2_score(y_eval, preds)
    rho, _ = spearmanr(y_eval, preds)

    print(f"[{desc}] MSE={mse:.4f}, R2={r2:.4f}, Spearman={rho:.4f}")
    return mse, rho, r2


if __name__ == "__main__":
    # Initialize device configuration
    device_config = DeviceConfig()
    device = device_config.device

    # Set gate set and dataset names to be used for this training run.
    search_space = 'ghz_b'
    dataset_name = 'train_dataset_augmented_gs2'
    model_name = f'{dataset_name}'  # Identifier for the current training configuration

    # Load path and model configurations.
    path_config = PathConfig()
    model_config = get_default_model_config_by_search_space(model_type="random_forest", search_space=search_space)

    # Set random seeds for reproducibility.
    random.seed(model_config.seed)
    np.random.seed(model_config.seed)
    torch.manual_seed(model_config.seed)

    # Construct the data path and load the dataset.
    data_path = os.path.join(path_config.paths['rf_data'], f'{dataset_name}.pt')
    data = load_data(data_path)
    print(f"Total data samples: {len(data)}")
    if len(data) == 0:
        raise ValueError("Data is empty.")

    # Split data into training, validation, and test sets.
    train_data, val_data, test_data = split.train_val_test_split(data, random_seed=model_config.seed)
    print(f"[INFO] Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    # Save test data for later use.
    save_data(os.path.join(path_config.paths['test_data'], f'test_rf_{search_space}.pt'), test_data)
    print(f"[INFO] Test Data saved to: {os.path.join(path_config.paths['test_data'])}, {len(test_data)}")

    # Print model training information.
    print(f"[INFO] Training: {model_name}")
    print(f"[INFO] Params: n_estimators={model_config.n_estimators}, max_depth={model_config.max_depth}, "
          f"seed={model_config.seed}, min_samples_split={model_config.min_samples_split}, "
          f"min_samples_leaf={model_config.min_samples_leaf}, max_features={model_config.max_features}, "
          f"n_jobs={model_config.n_jobs}")

    # Convert model configuration into a dictionary.
    config_dict = vars(model_config)

    # Train the Random Forest model.
    rf_model = train_model(config_dict, train_data)

    # Evaluate the model on the validation set.
    print("[INFO] Evaluating on Validation Set")
    val_mse, val_rho, val_r2 = evaluate_model(rf_model, val_data, desc="Validation")

    # Evaluate the model on the test set.
    print("[INFO] Evaluating on Test Set")
    test_mse, test_rho, test_r2 = evaluate_model(rf_model, test_data, desc="Testing")

    # Save the trained model to disk.
    model_save_path = os.path.join(path_config.paths['trained_models'], f'random_forest_{dataset_name}.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Model saved to {model_save_path}")

    # Prepare and save benchmark results.
    benchmark_results = {
        "model": model_name,
        "val_metrics": {"mse": val_mse, "spearman": val_rho, "r2": val_r2},
        "test_metrics": {"mse": test_mse, "spearman": test_rho, "r2": test_r2}
    }
    benchmark_results_path = os.path.join(path_config.paths['benchmark'], 'rf_benchmark_results.pkl')
    with open(benchmark_results_path, 'wb') as f:
        pickle.dump(benchmark_results, f)
    print(f"Benchmark results saved to {benchmark_results_path}")
