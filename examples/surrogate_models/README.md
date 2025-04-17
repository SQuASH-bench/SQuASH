# GCN Circuit Prediction

This script loads a pretrained GNN model to predict the fitness of a quantum circuit.

### Features
- Loads configuration from `config.json`
- Initializes and loads a pretrained model (e.g., `gcn_gs1`)
- Converts a Qiskit circuit to a data object and predicts its fitness

### Usage
1. **Check `config.json`** for model definitions, gate sets, and file names.
2. **Select `model_name`** (e.g., `gcn_gs1`) in the `__main__` block.
2. **Run:**
   ```bash
   python gcn_predict_circuit.py

The script loads the configuration and model, then evaluates the circuit and prints the predicted and ground truth performance.


# Random Forest Circuit Prediction

This script loads a pretrained Random Forest model to predict the fitness of a quantum circuit.

### Features
- Reads configuration (model file, gate sets) from `config.json`.
- Loads the selected Random Forest model (`random_forest_gs1`, `random_forest_gs2`, or `random_forest_ml`).
- Converts a Qiskit circuit into a feature tensor and predicts the circuit’s fitness.

### Usage
1. **Check `config.json`** for model definitions, gate sets, and file names.
2. **Select `model_name`** (e.g., `random_forest_gs1`) in the `__main__` block.
3. **Run**:
   ```bash
   python random_forest_predict_circuit.py

The script loads the model, the example circuit file (e.g., example_circuit_gs1.pkl), and prints the predicted and ground truth performance.