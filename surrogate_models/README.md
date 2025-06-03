# Quantum Circuit Regression Toolkit

This folder provides code for converting quantum circuits into graph representations, creating datasets, training models (Graph Neural Networks and Random Forests), and tuning hyperparameters. It is designed for users interested in quantum circuit regression tasks.

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

- **architectures/**  
  Contains model definitions such as the RegGNN and RandomForest.

- **prepare_dataset/**  
  Contains code to convert the raw data into graph format (input for RegGNN) or tensor format (input for RandomForest).
 
- **trained_models/**  
  Stores models trained from scratch along with their configurations.

- **tutorials/**  
  Includes tutorials to define, train and evaluate surrogate models to predict the performace of a PQC.

- **tuning/**  
  Contains hyperparameter tuning scripts for both GNN and Random Forest models using Optuna.

---



## Usage

⚙️ Configure Settings (Optional)

You can adjust the structure of surrogate models by modifying configuration in the `../config.py`. 

**What you can configure:**

- 🧱 **Model Hyperparameters**  
  Define architecture-specific parameters like embedding dimensions, number of layers, dropout ratios, etc.

- ⚙️ **Training Settings**  
  Customize batch size, learning rate, epochs, optimizer settings, and evaluation metrics.

- 🧬 **Gate Sets and Circuit Parameters**  
  Change the basis gate set, depth limits, or quantum circuit properties used for QAS.

- 📁 **Paths and Storage**  
  Set locations for saving models, logs, and generated data.



### Dataset Creation and Conversion
- Use utilities in the `../util` directory to convert quantum circuits (in QASM 3.0 format) to PyTorch Geometric Data objects.
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

Train and evaluate a GNN or Random Forest model using the provided code. Check out the folder `/tutorials`
to see the examples.

## Hyperparameter Tuning

Use the tuning scripts (in the `tuning/ directory`) to optimize model hyperparameters with Optuna.
Example:

```bash
python tune_gnn.py
python tune_rf.py
```