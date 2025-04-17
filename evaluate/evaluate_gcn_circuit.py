import os
import pickle
import torch

from surrogate_models.architectures.gnn.gnn_model import RegGNN
from util.qu_convert import qiskit_to_data_object

from config import DeviceConfig, get_model_config, PathConfig, QCConfig


def load_trained_model(model_name: str) -> RegGNN:
    """
    Load a pre-trained RegGNN model
    """
    device_config = DeviceConfig()
    device = device_config.device

    model_config = get_model_config(model_name, device)

    model = RegGNN(
        num_layer=model_config.layer_num,
        emb_dim=model_config.emb_dim,
        edge_attr_dim=model_config.qubit_num,
        num_node_features=model_config.num_node_features,
        drop_ratio=model_config.drop_ratio,
        graph_pooling=model_config.graph_pooling,
        JK=model_config.JK,
        freeze_gnn=False,
    )

    # construct the model file path and load state dictionary
    model_path = os.path.join(paths['la'], f'{model_name}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_circuit(circuit, model, gate_set=None, proxy=False) -> float:
    """
    Evaluate a quantum circuit using the pre-trained RegGNN model.

    This function converts a Qiskit circuit into a data object via the conversion utility,
    then uses the provided RegGNN model to perform inference on the circuit. The resulting
    prediction is returned as a floating point value representing the circuit's fitness or performance.

    Parameters:
    -----------
    circuit : Qiskit circuit

    model : RegGNN
        The pre-trained RegGNN model that will be used for the evaluation.
    gate_set : any
        A set of allowed gates or transformation rules to be used during circuit conversion.
        This parameter must be provided; if None, an Exception is raised.
    proxy : bool, default False
        A flag indicating whether to use a proxy transformation during circuit conversion.
        Defaults to False.

    Returns:
    --------
    float
        The predicted performance metric of the quantum circuit.

    """
    if gate_set is None:
        raise Exception("Error: The gate_set parameter cannot be None. Please provide a valid gate set.")

    # convert the Qiskit circuit to a compatible data object
    data_qc = qiskit_to_data_object(circuit, gate_set=gate_set, proxy=proxy)

    with torch.no_grad():
        prediction = model.predict(data_qc)

    return prediction.item()


if __name__ == "__main__":
    # retrieve configuration paths from PathConfig
    paths = PathConfig().paths

    # model identifier to be used for loading
    model_name = 'gs2_21_02'
    model = load_trained_model(model_name)

    # define the location of the pickle file containing the quantum circuit and its performance
    circuit_file = os.path.join(paths['rl_data'], 'logs_02_2024', 'unique_circuits', '0', 'qc_rl_754_7000.pkl')

    # load the circuit and its performance from the pickle file
    with open(circuit_file, 'rb') as file:
        loaded_data = pickle.load(file)
        circuit = loaded_data.get("initial_circuit")
        fidelity = loaded_data.get("fidelity")

    # evaluate the circuit using the loaded model and gate set from QCConfig
    prediction = evaluate_circuit(circuit, model, QCConfig().gate_set_ghz_b)
    print(f"Predicted performance for the circuit: {prediction}")
    print(f"Ground truth performance: {fidelity}")
