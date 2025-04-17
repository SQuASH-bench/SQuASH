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


def get_model_config(config_name, device='cpu'):
    """
    Retrieve the model configuration for a specific model name.
    """
    configs = {
        "gcn_ghz_a": {
            'device': device,
            'emb_dim': 1200,
            'layer_num': 8,
            'qubit_num': 3,
            'num_node_features': QCConfig().features_ghz_a,
            'drop_ratio': 0.012714767230404513,
            'batch_size': 32,
            'epochs': 3,
            'lr': 0.00042048670814195114,
            'decay': 1.2239395743425164e-06,
            'JK': 'mean',
            'seed': 42,
            'runseed': 42,
            'num_workers': 0,
            'patience': 7,
            'metric': 'spearman',
            'graph_pooling': 'max',
        },
        "gcn_ghz_b": {
            'device': device,
            'emb_dim': 1050,
            'layer_num': 8,
            'qubit_num': 3,
            'num_node_features': QCConfig().features_ghz_b,
            'drop_ratio': 0.0644893118913786,
            'batch_size': 128,
            'epochs': 100,
            'lr': 4.540520885756229e-05,
            'decay': 1.917208797826118e-06,
            'JK': 'mean',
            'seed': 42,
            'runseed': 42,
            'num_workers': 0,
            'patience': 7,
            'metric': 'spearman',
            'graph_pooling': 'attention',
        },
        "gcn_ls_a": {
            'device': device,
            'emb_dim': 1050,
            'layer_num': 8,
            'qubit_num': 4,
            'num_node_features': QCConfig().features_ghz_b,
            'drop_ratio': 0.0644893118913786,
            'batch_size': 32,
            'epochs': 100,
            'lr': 4.540520885756229e-05,
            'decay': 1.917208797826118e-06,
            'JK': 'mean',
            'seed': 42,
            'runseed': 42,
            'num_workers': 0,
            'patience': 7,
            'metric': 'spearman',
            'graph_pooling': 'attention',
        },
        "random_forest_ghz_a": {
            'device': device,
            'n_estimators': 350,
            'max_depth': 30,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'seed': 42,
            'runseed': 42,
            'batch_size': 64,
            'num_workers': 0,
            'metric': 'mse',
        },
        "random_forest_ghz_b": {
            'device': device,
            'n_estimators': 325,
            'max_depth': 30,
            'min_samples_split': 3,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'seed': 42,
            'runseed': 42,
            'batch_size': 64,
            'num_workers': 0,
            'metric': 'spearman',
        },
        "random_forest_ls_a": {
            'device': device,
            'n_estimators': 325,
            'max_depth': 30,
            'min_samples_split': 3,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'seed': 42,
            'runseed': 42,
            'batch_size': 64,
            'num_workers': 0,
            'metric': 'spearman',
        }
    }
    if config_name not in configs:
        raise ValueError(f"config '{config_name}' not found")
    return ModelConfig(**configs[config_name])


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
            'gcn_data': os.path.join(self.base_path, 'data/processed_data/', 'gcn_processed_data'),
            'rf_data': os.path.join(self.base_path, 'data', 'rf_data'),
            'test_data': os.path.join(self.base_path, 'data', 'test_data'),
            'trained_models': os.path.join(self.base_path, 'results', 'trained_models'),
            'benchmark': os.path.join(self.base_path, 'results', 'benchmark'),
            'plots': os.path.join(self.base_path, 'results', 'plots'),
        }
        print(self.paths['gcn_data'])
