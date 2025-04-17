# Quantum Circuit Regression Toolkit

This repository provides code for converting quantum circuits into graph representations, creating datasets, training models (Graph Neural Networks and Random Forests), and tuning hyperparameters. It is designed for users interested in quantum circuit regression tasks.

---

## Overview

- **Graph Conversion & Dataset Creation:**  
  Convert OPENQASM 3.0 circuits into PyTorch Geometric `Data` objects. These objects include node features (e.g., one-hot encoded gate types, parameters), edge indices, and edge attributes representing qubit interactions. Functions to visualize the resulting graphs are also provided.

- **Model Training & Evaluation:**  
  Train and evaluate regression models, primarily using a Graph Neural Network (RegGNN) and an optional Random Forest surrogate. The training scripts include early stopping and loss/metric tracking over epochs.

- **Hyperparameter Tuning:**  
  Use Optuna to tune model hyperparameters for both GNN and Random Forest models, with configurable metrics (e.g., Spearman's rho, MSE, R²).

- **Configuration Management:**  
  All settings for devices, quantum circuit parameters, and model hyperparameters are managed via configuration classes (`DeviceConfig`, `QCConfig`, `ModelConfig`, and `PathConfig`).

---

## Directory Structure

- **config/**  
  Contains configuration settings for devices, quantum circuit parameters, model hyperparameters, and file paths.

- **model/**  
  Contains model definitions such as the RegGNN used for quantum circuit regression.

- **util/**  
  Includes:
  - **Data Loader & Dataset Creation:** Functions to load and split data, create dataset objects from circuit files or a database, and save/load processed data.
  - **Quantum Circuit Conversion:** Utilities to convert circuits (QPY/QASM) to graph representations (e.g., `process_graph`, `circuit_to_tensor`, `visualize_graph`).
  - **Miscellaneous Utility Functions:** Functions for hyperparameter tuning, benchmark result saving, etc.

- **tuning/**  
  Contains hyperparameter tuning scripts for both GNN and Random Forest models using Optuna.

---



## Usage

### Dataset Creation and Conversion
- Use utilities in the `util` directory to convert quantum circuits (in QASM 3.0 format) to PyTorch Geometric Data objects.
- Example:  
  ```python
  from config import QCConfig
  from util.qu_convert import process_graph
  
  qc_config = QCConfig()
  qasm_example = """
  OPENQASM 3.0;
  include "stdgates.inc";
  qubit[3] q;
  h q[0];
  rx(1.57) q[1];
  ry(1.57) q[2];
  cx q[0], q[1];
  id q[2];
  """
  data = process_graph(qasm_example, fidelity=0.5, gate_set=qc_config.gate_set_1)
  print(data)



## Training a Model

Train a GNN or Random Forest model using the provided training scripts.

**Example (for Random Forest):**
```bash
python surrogate_gcn.py
```

Similarly, use the GNN training script (surrogate_rf.py) to train a RandomForestRegressor model.


## Hyperparameter Tuning

Use the tuning scripts (in the `tuning/ directory`) to optimize model hyperparameters with Optuna.
Example:

```bash
python tune_gnn.py
python tune_rf.py
```

## Evaluating and Visualizing Results

The training scripts record `MSE`, `Spearman's rho`, and `R2` over epochs. Use the provided plotting functions to visualize performance metrics.