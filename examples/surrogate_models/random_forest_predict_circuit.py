import json
import pickle

from util.rf_convert import circuit_to_tensor


def load_config(config_file="config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def load_model(model_name="random_forest_gs1", config=None):
    """
    Load the random forest model using configuration parameters.

    Args:
        model_name (str): Either "random_forest_gs1" or "random_forest_gs2" to select the model.
        config (dict): Dictionary with configuration loaded from JSON.

    Returns:
        model: The loaded random forest model - RandomForestRegressor from sklearn.
    """
    if config is None:
        raise ValueError("Configuration must be provided.")

    model_config = config.get(model_name)
    if model_config is None:
        raise ValueError(f"Model configuration for '{model_name}' not found in config.")

    # Use the model file specified in the config, or default to <model_name>.pkl
    model_file = model_config.get("model_file", f"{model_name}.pkl")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_circuit(circuit, model, fidelity, gate_set=None):
    """
    Evaluate the fitness of a circuit using the provided random forest model.

    Args:
        circuit: The circuit object.
        model: The loaded random forest model.
        gate_set (list): List of gate strings to use for conversion.
        fidelity: Fidelity value from the circuit file used in conversion.

    Returns:
        float: The predicted fitness.
    """
    if gate_set is None:
        print("Gate set is not specified, setting it to ['cx', 'h', 'rx', 'ry', 'rz', 'id']")
        gate_set = ["cx", "h", "rx", "ry", "rz", "id"]
    data_qc = circuit_to_tensor(qasm3=circuit, fidelity=fidelity, gate_set=gate_set, num_qubits=4)
    X, _ = data_qc

    if hasattr(X, "cpu"):
        X = X.cpu().numpy()
    X = X.reshape(1, -1)

    prediction_array = model.predict(X)
    return float(prediction_array[0])


def get_circuit_file_and_gate_set(model_name: str):
    if model_name == "random_forest_gs1":
        gate_set = config.get("gates_ghz_a")
        circuit_file = '../../random_forest/example_circuit_ghz_a.pkl'
    elif model_name == "random_forest_gs2":
        gate_set = config.get("gates_ghz_b")
        circuit_file = '../../random_forest/example_circuit_ghz_b.pkl'
    elif model_name == "random_forest_ml":
        gate_set = config.get("gates_ls_a")
        circuit_file = '../../random_forest/example_circuit_ml.pkl'
    else:
        raise ValueError("Unknown model name provided.")
    return circuit_file, gate_set


if __name__ == "__main__":
    config = load_config("../../surrogate_models/pretrained/random_forest/config.json")

    model_name = "random_forest_ml"
    model = load_model(model_name, config=config)

    circuit_file, gate_set = get_circuit_file_and_gate_set(model_name)

    with open(circuit_file, 'rb') as file:
        loaded_data = pickle.load(file)
        circuit = loaded_data.get("initial_circuit")
        train_accuracy = loaded_data.get("train_acc")
        fidelity = loaded_data.get("fidelity")

    if model_name == 'random_forest_ml':
        prediction = evaluate_circuit(circuit, model, train_accuracy, gate_set)
        print(f"Predicted performance for the circuit: {prediction}")
        print(f"Ground truth: {train_accuracy}")
    else:
        prediction = evaluate_circuit(circuit, model, fidelity, gate_set)
        print(f"Ground truth: {fidelity}")
