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

import json
import os
import torch

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DeviceConfig:
    """
    Configuration for selecting the computing device.
    """
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class QCConfig:
    """
    Quantum Circuit Configuration.

    Attributes:
        max_param_gate (int): Maximum number of parameters for a gate.
        gate_set_ghz_a (list): Allowed gates for configuration 1, e.g., ['cx', 'h', 'rx', 'ry', 'rz', 'id'].
        features_ghz_a (int): Number of features for gate set 1.
        gate_set_ghz_b (list): Allowed gates for configuration 2, e.g., ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x'].
        features_ghz_b (int): Number of features for gate set 2.
        gate_set_ghz_a (list): Allowed gates for machine learning models.
        features_ls_a (int): Number of features for the ML gate set.
    """
    max_param_gate: int = 1
    gate_set_ghz_a: list = field(default_factory=lambda: ['cx', 'h', 'rx', 'ry', 'rz', 'id'])
    features_ghz_a: int = 7
    gate_set_ghz_b: list = field(default_factory=lambda: ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x'])
    features_ghz_b: int = 8
    gate_set_ls_a: list = field(default_factory=lambda: ['cx', 'h', 'rx', 'ry', 'swap', 'crx', 'cry'])
    features_ls_a: int = 8


@dataclass
class ModelConfig:
    """
    Configuration for model hyperparameters.
    """
    device: torch.device
    seed: int
    runseed: int
    batch_size: int
    num_workers: int
    epochs: Optional[int] = None
    emb_dim: Optional[int] = None
    layer_num: Optional[int] = None
    qubit_num: Optional[int] = None
    num_node_features: Optional[int] = None
    drop_ratio: Optional[float] = None
    lr: Optional[float] = None
    decay: Optional[float] = None
    JK: Optional[str] = None
    patience: Optional[int] = None
    metric: Optional[str] = None
    graph_pooling: Optional[str] = None
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    random_state: Optional[int] = None
    optuna_trials: Optional[int] = None
    min_samples_split: Optional[int] = None
    min_samples_leaf: Optional[int] = None
    max_features: Optional[str] = None
    n_jobs: Optional[int] = None


def get_model_config_from_path(model_config_path, device):
    try:
        with open(model_config_path, 'r') as f:
            config = json.load(f)
            config["device"] = device
            del config["PATHS"]
            del config["timestamp"]
        return ModelConfig(**config)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load model config from '{model_config_path}': {e}")


def get_default_model_config_by_search_space(model_type, search_space, features=None, device='cpu'):
    """
    Retrieve the model configuration for a specific model name.

    Args:
        model_config_path (str): Path to the saved JSON config file.
        model_name (str): Name of the model (used as a key in the config).
        search_space (str): Name of the benchmark search space.
        features (int): Number of input node features.
        device (str): 'cpu' or 'cuda'.

    Returns:
        ModelConfig: A ModelConfig instance with updated parameters.
    """
    try:
        config = _get_default_model_config(search_space, model_type, features, device)
        config['device'] = device
        config['num_node_features'] = features
        config.setdefault('seed', 42)
        config.setdefault('runseed', 42)
        config.setdefault('num_workers', 0)
        config.setdefault('patience', 7)
        return ModelConfig(**config)
    except Exception as e:
        raise RuntimeError(f"Error processing model config: {e}")


def _get_default_model_config(search_space, model_type, features=None, device="cpu"):
    """
    Fallback config based on model_name and search_space.
    """
    default_configs = {
        "gcn_ghz_a": {
            'emb_dim': 1200,
            'layer_num': 8,
            'qubit_num': 3,
            'num_node_features': features,
            'drop_ratio': 0.012714767230404513,
            'batch_size': 32,
            'epochs': 3,
            'lr': 0.00042048670814195114,
            'decay': 1.2239395743425164e-06,
            'JK': 'mean',
            'graph_pooling': 'max',
            'metric': 'spearman',
        },
        "gcn_ghz_b": {
            'emb_dim': 1050,
            'layer_num': 8,
            'qubit_num': 3,
            'num_node_features': features,
            'drop_ratio': 0.0644893118913786,
            'batch_size': 128,
            'epochs': 100,
            'lr': 4.540520885756229e-05,
            'decay': 1.917208797826118e-06,
            'JK': 'mean',
            'graph_pooling': 'attention',
            'metric': 'spearman',
        },
        "gcn_ls_a": {
            'emb_dim': 1050,
            'layer_num': 8,
            'qubit_num': 4,
            'num_node_features': features,
            'drop_ratio': 0.0644893118913786,
            'batch_size': 32,
            'epochs': 100,
            'lr': 4.540520885756229e-05,
            'decay': 1.917208797826118e-06,
            'JK': 'mean',
            'graph_pooling': 'attention',
            'metric': 'spearman',
        },
        "random_forest_ghz_a": {
            'n_estimators': 350,
            'max_depth': 30,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'batch_size': 64,
            'metric': 'mse',
        },
        "random_forest_ghz_b": {
            'n_estimators': 325,
            'max_depth': 30,
            'min_samples_split': 3,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'batch_size': 64,
            'metric': 'spearman',
        },
        "random_forest_ls_a": {
            'n_estimators': 325,
            'max_depth': 30,
            'min_samples_split': 3,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'batch_size': 64,
            'metric': 'spearman',
        }
    }

    if model_type == "gcn":
        fallback_key = f"gcn_{search_space}"
    elif model_type == "random_forest":
        fallback_key = f"random_forest_{search_space}"
    else:
        raise KeyError(f"Unknown model type for fallback: '{model_type}'")
    config = default_configs[fallback_key].copy()
    config["device"] = device
    if fallback_key not in default_configs:
        raise KeyError(f"No default config found for '{fallback_key}'")

    return default_configs[fallback_key].copy()


@dataclass
class PathConfig:
    """
    Configuration for project file paths.
    """
    base_path: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    paths: Dict[str, str] = field(init=False)

    def __post_init__(self):
        # Define subdirectories for tuning studies, datasets, models, benchmarks, and plots.
        self.paths = {
            'optuna_studies': os.path.join(self.base_path, 'surrogate_models/tuning', 'studies'),
            'raw_data': os.path.join(self.base_path, 'data/raw_data/'),
            'gcn_data': os.path.join(self.base_path, 'data/processed_data/', 'gcn_processed_data'),
            'rf_data': os.path.join(self.base_path, 'data', 'rf_data'),
            #            'test_data': os.path.join(self.base_path, 'data', 'test_data'),
            'trained_models': os.path.join(self.base_path, 'surrogate_models', 'trained_models'),
            'benchmark': os.path.join(self.base_path, 'benchmark', 'results'),
            #            'plots': os.path.join(self.base_path, 'benchmark/results', 'plots'),
        }
