import os

from config import QCConfig
from util.config_utils import get_gate_set_and_features_by_name
from util.data_loader import save_data, get_data
from util.qu_convert import process_graph
from util.rf_convert import circuit_to_tensor


def _process_common_args(circuit, gate_set_name):
    """Helper to extract shared processing values."""
    qu_circuit = circuit.get("init_circuit")
    init_params = None
    fidelity = circuit.get("fidelity", circuit.get("train_accuracy"))
    num_qubits = 4 if 'ls_a' in gate_set_name else 3
    return qu_circuit, init_params, fidelity, num_qubits


def create_gcn_data(path, gate_set, gate_set_name, proxy=False):
    """Create GCN-compatible PyG data objects."""
    data = []
    for folder in os.listdir(path):
        subfolder_path = os.path.join(path, folder)
        if not os.path.isdir(subfolder_path):
            continue
        qiskit_circuits = get_data(subfolder_path)
        for circuit in qiskit_circuits:
            try:
                qu_circuit, init_params, fidelity, num_qubits = _process_common_args(
                    circuit, gate_set_name
                )
                processed = process_graph(
                    qu_circuit,
                    init_params=init_params,
                    fidelity=fidelity,
                    gate_set=gate_set,
                    proxy=proxy,
                    num_qubits=num_qubits
                )
                data.append(processed)
            except Exception as e:
                print(f"Error processing circuit {circuit}: {e}")
    return data


def create_rf_data(path, gate_set, gate_set_name):
    """Create RandomForest-compatible tensor data."""
    data = []
    for folder in os.listdir(path):
        subfolder_path = os.path.join(path, folder)
        if not os.path.isdir(subfolder_path):
            continue
        qiskit_circuits = get_data(subfolder_path)
        for circuit in qiskit_circuits:
            try:
                qu_circuit, _, fidelity, num_qubits = _process_common_args(
                    circuit, gate_set, gate_set_name
                )
                processed = circuit_to_tensor(qu_circuit, fidelity, gate_set, num_qubits)
                data.append(processed)
            except Exception as e:
                print(f"Error processing circuit {circuit}: {e}")
    return data


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load config and set variables
    qc_config = QCConfig()
    data_name = 'demo_dataset_ghz_a'
    gate_set_name = 'gate_set_ghz_a'
    model = 'GCN'  # 'GCN' or 'RandomForest'
    proxy = False

    gate_set, _ = get_gate_set_and_features_by_name(gate_set_name)
    input_path = os.path.abspath(os.path.join(base_dir, '../../data/raw_data', data_name))

    if model == 'GCN':
        output_path = os.path.abspath(
            os.path.join(base_dir, '../../data/processed_data/gcn_processed_data', f'{data_name}.pt')
        )
        data = create_gcn_data(path=input_path, gate_set=gate_set, gate_set_name=gate_set_name, proxy=proxy)
    elif model == 'RandomForest':
        output_path = os.path.abspath(
            os.path.join(base_dir, '../../data/processed_data/rf_processed_data', f'{data_name}.pt')
        )
        data = create_rf_data(input_path, gate_set, gate_set_name)
    else:
        raise ValueError('Unknown model')

    print(f"Data processing completed. Saving to: {output_path}")
    save_data(output_path, data)
