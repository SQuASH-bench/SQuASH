import os

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Parameter
from qiskit.circuit.library import IGate
from qiskit_aer import Aer

from config import get_model_config_from_path, PathConfig
from evaluate.evaluate_utils import load_gnn_model
from util.qu_convert import qiskit_to_data_object
from qiskit.qasm3 import loads


def load_gnn_benchmark_model(model_name, search_space, model_config=None, device='cpu'):
    """
    Initialize and load the model using configuration parameters.

    Args:
        model_name (str): Either "gcn_gs1" or "old_gs2" to select the model.
        device (str): Device to load the model on.
        config (dict): Dictionary with configuration loaded from JSON.

    Returns:
        :param model_name: the name of the model, the models are stored under ../surrogate_models/
        :param device: cpu or gpu
        :param model_config: json config for the gnn model
        :param search_space: ghz_a, ghz_b, or ls_a
    """
    path_config = PathConfig()
    if model_config is None:
        model_config_path = os.path.join(path_config.paths['benchmark_search_spaces'],
                                         f'{search_space}\\surrogate_models\\configs',
                                         f'{model_name}_config.json')
        model_config = get_model_config_from_path(model_config_path, device)
    if model_config is None:
        raise ValueError(f"Model configuration for '{model_name}' not found in config.")

    model_path = os.path.join(path_config.paths['benchmark_search_spaces'], f"{search_space}\\surrogate_models",
                              f"{model_name}.pth")
    gnn_model = load_gnn_model(model_path, model_config)

    gnn_model.to(device)
    gnn_model.eval()
    return gnn_model


def predict_circuit_performance(circuit, model, gate_set):
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


def convert_qasm_circuit_into_trainable_pqc(qasm_str):
    qc = loads(qasm_str)

    # print("____________________Initial QC__________________________", qc.draw())
    new_data = []
    params = []
    for gate in qc.data:
        if gate.operation.name == 'u':
            new_instruction = CircuitInstruction(IGate(), gate.qubits)  # ID gate as CircuitInstruction
            new_data.append(new_instruction)
        else:
            new_data.append(gate)
    qc_new_1 = QuantumCircuit(3)
    for gate in new_data:
        qc_new_1.append(gate.operation, gate.qubits)
    # print("____________________QC Converted__________________________", qc_new_1.draw())

    qc_new_2 = QuantumCircuit(3)
    param_index = 0

    for instruction in qc_new_1.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        if operation.params:
            new_params = []
            for p in operation.params:
                params.append(p)
                param = Parameter(f'θ{param_index}')
                new_params.append(param)  # Use the Parameter object
                param_index += 1

            qc_new_2.append(operation.__class__(param), qubits, clbits)
        else:
            qc_new_2.append(operation, qubits, clbits)


    return qc_new_2, params
