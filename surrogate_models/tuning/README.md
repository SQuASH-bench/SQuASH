# Model Tuning Scripts

This folder contains scripts for hyperparameter optimization using [Optuna](https://optuna.org/). They are designed to tune regression models for quantum circuit evaluation:

- **Graph Neural Network (RegGNN) Tuning**: `tune_gnn.py`
- **Random Forest Tuning**: `tune_rf.py`

---

## Overview

### 1. GNN Tuning (`tune_gnn.py`)
- **Purpose**: Hyperparameter search for a Graph Neural Network (RegGNN) model.
- **Search space includes**:  
  - Embedding dimension (`emb_dim`)
  - Number of GNN layers (`layer_num`)
  - Dropout rate (`drop_ratio`)
  - Pooling method (`graph_pooling`)
  - Learning rate (`lr`)
  - Weight decay (`decay`)
  - Jumping Knowledge (`JK`)
- **Optimization metric**: Choose between mean squared error (`mse`) or Spearman correlation (`spearman`), set in config.
- **Early stopping**: Stops tuning if no improvement over a defined patience.
- **Optuna study**: Results are saved in a SQLite database for later analysis.

### 2. Random Forest Tuning (`tune_rf.py`)
- **Purpose**: Hyperparameter search for a RandomForestRegressor.
- **Search space includes**:  
  - Number of trees (`n_estimators`)
  - Maximum tree depth (`max_depth`)
  - Minimum samples to split/leaf
  - Feature selection method (`max_features`)
- **Optimization metric**: Choose `mse` or `spearman` in config.
- **Parallel execution**: All CPU cores are used.
- **Optuna study**: Results saved to SQLite.

---

## Data & Configuration

1. **Prepare your dataset**  
   - For GNN: Should be a `.pt` file containing a list of graph data objects.
   - For RF: Should be a `.pt` file containing a list of tuples `(X, y)` or `(X, y)` arrays/tensors.
   - Place your data in directories specified by your configuration, e.g.:
     - `config['PATHS']['gcn_data']` for GNN
     - `config['PATHS']['rf_data']` for RF

2. **Adjust configuration**  
   - Configuration is loaded via helper functions (see `prepare_paths_and_config`, `get_default_model_config_by_search_space`).
   - Set your random seed, metric (`'mse'` or `'spearman'`), and other options in your config.
   - Update batch size, number of workers, and path locations as needed.

---

## Configuration Tips

- **Metric**:  
  - Use `'mse'` for error minimization or `'spearman'` for correlation maximization.
  - Set this in your config dictionary.

- **Seeds & Reproducibility**:  
  - Both scripts call `set_seed` with a run-specific seed from the config for deterministic results.

- **Early stopping (GNN)**:  
  - Controlled via the `patience` parameter in your config.

---

## Results

- After running, best hyperparameters are printed to the terminal.
- All Optuna study results are saved in SQLite DB files under your Optuna studies directory (see config).

