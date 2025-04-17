import json
import torch
import pickle

from surrogate_models.architectures.gnn.gnn_model import RegGNN
from util.qu_convert import qiskit_to_data_object


def load_config(config_file="config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def load_model(model_name='gcn_gs1', device='cpu', config=None):
    """
    Initialize and load the model using configuration parameters.

    Args:
        model_name (str): Either "gcn_gs1" or "old_gs2" to select the model.
        device (str): Device to load the model on.
        config (dict): Dictionary with configuration loaded from JSON.

    Returns:
        model: The loaded RegGNN model.
    """
    if config is None:
        raise ValueError("Configuration must be provided.")

    model_config = config.get(model_name)
    if model_config is None:
        raise ValueError(f"Model configuration for '{model_name}' not found in config.")

    model = RegGNN(
        num_layer=model_config["layer_num"],
        emb_dim=model_config["emb_dim"],
        edge_attr_dim=model_config["qubit_num"],
        num_node_features=model_config["num_node_features"],
        drop_ratio=model_config["drop_ratio"],
        graph_pooling=model_config["graph_pooling"],
        JK=model_config["JK"],
        freeze_gnn=False,
    )

    model.load_state_dict(torch.load(f'{model_name}.pth', map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_circuit(circuit, model, gate_set):
    """
    Evaluate the fitness of a circuit using the provided model.

    Args:
        circuit: The circuit object.
        model: The loaded model.
        gate_set (list): List of gate strings to use for conversion.

    Returns:
        float: The predicted fitness.
    """
    data_qc = qiskit_to_data_object(circuit=circuit, gate_set=gate_set, num_qubits=model.edge_attr_dim)
    with torch.no_grad():
        prediction = model.predict(data_qc)
    return prediction.item()


def get_circuit_file_and_gate_set(model_name: str):
    if model_name == "gcn_ghz_a":
        gate_set = config.get("gates_ghz_a")
        circuit_file = 'example_circuit_ghz_a.pkl'
    elif model_name == "gcn_ghz_b":
        gate_set = config.get("gates_ghz_b")
        circuit_file = 'example_circuit_ghz_b.pkl'
    elif model_name == "gcn_ls_a":
        gate_set = config.get("gates_ls_a")
        circuit_file = 'example_circuit_ls_a.pkl'
    else:
        raise ValueError("Unknown model name provided.")
    return circuit_file, gate_set


if __name__ == "__main__":
    config = load_config("../../surrogate_models/pretrained/gcn/config.json")

    model_name = "gcn_ml"
    model = load_model(model_name, device="cpu", config=config)
    circuit_file, gate_set = get_circuit_file_and_gate_set(model_name)

    with open(circuit_file, 'rb') as file:
        loaded_data = pickle.load(file)
        circuit = loaded_data.get("initial_circuit")
        fidelity = loaded_data.get("fidelity")
        train_accuracy = loaded_data.get("train_acc")

    prediction = evaluate_circuit(circuit, model, gate_set)
    print(f"Predicted performance for the circuit: {prediction}")
    if model_name == 'gcn_ml':
        print(f"Ground truth: {train_accuracy}")
    else:
        print(f"Ground truth: {fidelity}")
