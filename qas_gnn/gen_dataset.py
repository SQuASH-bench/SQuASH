import os

from config import QCConfig
from util.data_loader import create_data, save_data

"""
Main script for creating and saving a quantum circuit dataset.

This script uses the configuration defined in QCConfig and dataset utility functions to:
  - Select the appropriate gate set based on a given configuration (gs1, gs2, or ml).
  - Choose the model type (RandomForest or GCN) and set file paths accordingly.
  - Process input circuit data from an input directory into PyTorch Geometric Data objects.
  - Save the processed dataset to a specified output path for later training or evaluation.

Optionally, users may choose to load data from a database using create_data_from_database (currently commented).
"""


if __name__ == '__main__':
    # Load quantum circuit configuration
    qc_config = QCConfig()

    # Set dataset and model identifiers
    data_name = 'test_dataset_ghz_a'
    gate_set_name = 'gs1'
    model = 'GCN'  # Options: 'RandomForest' or 'GCN'
    proxy = False

    # Select the appropriate gate set from the configuration based on gate_set_name.
    if gate_set_name == 'gs1':
        gs = qc_config.gate_set_1
    elif gate_set_name == 'gs2':
        gs = qc_config.gate_set_2
    elif gate_set_name == 'ml':
        gs = qc_config.gate_set_ml
    else:
        raise ValueError(f"Unknown gate set name: {gate_set_name}")

    # Determine the base directory of the current script.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Set input and output paths based on the chosen model type.
    if model == 'GCN':
        # For GCN, the data is saved in the 'test_data' directory.
        input_path = f"your/path/{data_name}"
        output_path = os.path.join(base_dir, 'data', 'test_data', f'{data_name}.pt')
    elif model == 'RandomForest':
        # For RandomForest, the data is saved in the 'rf_data' directory.
        input_path = f"your/path/{data_name}"
        output_path = os.path.join(base_dir, 'data', 'rf_data', f'{data_name}.pt')
    else:
        raise ValueError('Unknown model')

    # Create the dataset from the input path.
    # This function processes quantum circuit files into PyTorch Geometric Data objects.
    data = create_data(input_path, gs, model, proxy=proxy, gate_set_name=gate_set_name)

    # Optionally, data can be generated from a database:
    # data = create_data_from_database(os.path.join(base_dir, 'data', 'database', 'unique_circuits_gesamt_.db'), gs, model)

    # Save the processed dataset to the output path.
    save_data(output_path, data)

    # Additional examples, commented-out:
    # The following code shows how to load individual circuits,
    # create a QuantumCircuit object, and visualize the graph representation.
    #
    # qc = []
    # path = os.path.join(base_dir, 'data/merged_data/gs2_21_02/')
    # for file_name in os.listdir(path):
    #     file_path = os.path.join(path, file_name)
    #     if file_name.endswith('.pkl'):
    #         with open(file_path, 'rb') as file:
    #             qiskit_circuits = pickle.load(file)
    #         initial_circuit = qiskit_circuits.get("initial_circuit")
    #         circuit = loads(initial_circuit)
    #         qc.append(circuit)
    #
    # Example usage with QuantumCircuit drawing and processing:
    #
    # qc1 = QuantumCircuit(3, 3)
    # qc1.h(0)
    # qc1.rx(1.57, 1)
    # qc1.cx(1, 2)
    # qc1.cx(0, 1)
    # qc1.measure([0, 1, 2], [0, 1, 2])
    # qc1.draw('mpl')
    # qc1.data = [op for op in qc1.data if op[0].name != 'measure']
    # print(qc1)
    #
    # qasm = dumps(qc1)
    # circuit = process_graph(qasm, 0.5, qc_config.gate_set_1)
    # print(circuit)
    # print("Node features (x):")
    # print(circuit.x)
    # print("\nEdge index:")
    # print(circuit.edge_index)
    # print("\nEdge attributes (edge_attr):")
    # print(circuit.edge_attr)
    # print("\nFitness value (y):")
    # print(circuit.y)
    # visualize_graph(circuit, qc_config.gate_set_1)
