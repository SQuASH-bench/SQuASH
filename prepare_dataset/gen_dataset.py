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


def get_gate_set_by_name(gate_set_name):
    if gate_set_name == 'gate_set_ghz_a':
        gs = qc_config.gate_set_ghz_a
    elif gate_set_name == 'gate_set_ghz_b':
        gs = qc_config.gate_set_ghz_b
    elif gate_set_name == 'gate_set_ls_a':
        gs = qc_config.gate_set_ls_a
    else:
        raise ValueError(f"Unknown gate set name: {gate_set_name}")
    return gs


if __name__ == '__main__':
    # Load quantum circuit configuration
    qc_config = QCConfig()

    # Set dataset and model identifiers
    data_name = 'test_dataset_ghz_a'
    gate_set_name = 'gate_set_ghz_a'
    model = 'GCN'  # Options: 'RandomForest' or 'GCN'
    proxy = False

    # Select the appropriate gate set from the configuration based on gate_set_name.
    if gate_set_name == 'gate_set_ghz_a':
        gs = qc_config.gate_set_ghz_a
    elif gate_set_name == 'gate_set_ghz_b':
        gs = qc_config.gate_set_ghz_b
    elif gate_set_name == 'gate_set_ls_a':
        gs = qc_config.gate_set_ls_a
    else:
        raise ValueError(f"Unknown gate set name: {gate_set_name}")

    # Determine the base directory of the current script.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Set input and output paths based on the chosen model type.
    if model == 'GCN':
        # For GCN, the data is saved in the 'test_data' directory.
        input_path = os.path.abspath(os.path.join(base_dir, '../data', 'raw_data', f'{data_name}'))
        output_path = os.path.abspath(
            os.path.join(base_dir, '../data/processed_data/', 'gcn_processed_data', f'{data_name}.pt'))
    elif model == 'RandomForest':
        # For RandomForest, the data is saved in the 'rf_data' directory.
        input_path = os.path.abspath(os.path.join(base_dir, '../data', 'raw_data', f'{data_name}'))
        output_path = os.path.abspath(
            os.path.join(base_dir, '../data/processed_data/', 'rf_processed_data', f'{data_name}.pt'))
    else:
        raise ValueError('Unknown model')

    # Create the dataset from the input path.
    # This function processes quantum circuit files into PyTorch Geometric Data objects.
    print(f"Start processing data from the path: {input_path}")
    data = create_data(input_path, gs, model, proxy=proxy, gate_set_name=gate_set_name)

    # Optionally, data can be generated from a database:
    # data = create_data_from_database(os.path.join(base_dir, 'data', 'database', 'unique_circuits_gesamt_.db'), gs, model)

    save_data(output_path, data)
