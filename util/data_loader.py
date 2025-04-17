import torch
import pickle
import os
import sqlite3

import numpy as np

from util.qu_convert import process_graph
from util.rf_convert import circuit_to_tensor


def get_data(path):
    """
    Load circuit data from all pickle files in a given directory.

    This function iterates over all files in the specified directory and loads
    any file ending with '.pkl'. It extracts relevant information such as the initial circuit,
    optimal circuit, fidelity, and training accuracy, then appends these as a dictionary.

    Args:
        path (str): The directory path containing pickle files.

    Returns:
        list: A list of dictionaries, each containing circuit data.
    """
    circuit = []
    # Iterate over files in the directory.
    for file_name in os.listdir(path):
        if file_name.endswith('.pkl'):
            full_path = os.path.join(path, file_name)
            try:
                with open(full_path, 'rb') as file:
                    loaded_data = pickle.load(file)

                initial_circuit = loaded_data.get("initial_circuit")
                optimal_circuit = loaded_data.get("optimal_circuit")
                fidelity = loaded_data.get("fidelity")
                train_accuracy = loaded_data.get("train_acc")

                circuit.append({
                    "init_circuit": initial_circuit,
                    "opt_circuit": optimal_circuit,
                    "fidelity": fidelity,
                    "train_accuracy": train_accuracy,
                })

            except FileNotFoundError:
                print(f"File not found: {full_path}")
            except Exception as e:
                print(f"Error loading file {full_path}: {e}")

    return circuit


def create_data(path, gate_set=None, model=None, proxy=False, gate_set_name=None):
    """
    Process circuit data from subdirectories and convert them into data objects
    appropriate for a given model (GCN or RandomForest).

    For each subfolder in the provided path, it loads the circuit data using `get_data`
    and then processes each circuit using the specified conversion function:
      - For GCN models: uses `process_graph` from `util.qu_convert`
      - For RandomForest models: uses `circuit_to_tensor` from `util.rf_convert`

    The number of qubits used in processing depends on the gate_set_name (e.g., 'ml' uses 4 qubits).

    Args:
        path (str): Path containing subfolders with circuit pickle files.
        gate_set (list, optional): List of gate strings to use for processing. Defaults to ['h', 'cx'].
        model (str, required): The model type; either 'GCN' or 'RandomForest'.
        proxy (bool, optional): Whether to use proxy information in processing. Defaults to False.
        gate_set_name (str, optional): Additional identifier to choose proper processing (e.g., 'ml').

    Returns:
        list: A list of processed data objects.
    """
    if gate_set is None:
        print("Gate set is unspecified, setting to ['h', 'cx']")
        gate_set = ['h', 'cx']
    data = []
    # Process each subfolder.
    for folder in os.listdir(path):
        subfolder_path = os.path.join(path, folder)
        if os.path.isdir(subfolder_path):
            qiskit_circuits = get_data(subfolder_path)
            for circuit in qiskit_circuits:
                try:
                    # Get the initial circuit data.
                    qu_circuit = circuit.get("init_circuit")
                    # init_params might be provided; if not, use None.
                    init_params = None
                    fidelity = circuit.get("fidelity")
                    if fidelity is None:
                        fidelity = circuit.get("train_accuracy")

                    # Process circuit based on model type and gate set name.
                    if model == 'GCN':
                        if gate_set_name == 'ml':
                            processed_circuit = process_graph(qu_circuit, init_params=init_params, fidelity=fidelity,
                                                              gate_set=gate_set, proxy=proxy, num_qubits=4)
                        else:
                            processed_circuit = process_graph(qu_circuit, init_params=init_params, fidelity=fidelity,
                                                              gate_set=gate_set, proxy=proxy, num_qubits=3)
                        data.append(processed_circuit)
                    elif model == 'RandomForest':
                        if gate_set_name == 'ml':
                            processed_circuit = circuit_to_tensor(qu_circuit, fidelity, gate_set, num_qubits=4)
                        else:
                            processed_circuit = circuit_to_tensor(qu_circuit, fidelity, gate_set, num_qubits=3)
                        data.append(processed_circuit)
                    else:
                        raise ValueError(f"Unknown model: {model}")
                except Exception as e:
                    print(f"Error processing circuit {circuit}: {e}")

    return data


def create_data_from_database(db_path="dataset.db", gate_set=None, model=None):
    """
    Create processed data by reading circuit information from a SQLite database.

    This function connects to the specified SQLite database, retrieves circuit information
    (circuit string, fidelity, and initial parameters), and processes each circuit using:
      - `process_graph` for GCN models
      - `circuit_to_tensor` for RandomForest models

    Args:
        db_path (str, optional): Path to the SQLite database file. Defaults to "dataset.db".
        gate_set (list, optional): List of gate strings for processing. Defaults to ['h', 'cx'].
        model (str, required): The model type; either 'GCN' or 'RandomForest'.

    Returns:
        list: A list of processed data objects.
    """
    if gate_set is None:
        print("Gate set is unspecified, setting to ['h', 'cx']")
        gate_set = ['h', 'cx']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT circuit, fidelity, initial_params FROM circuits")
    rows = cursor.fetchall()
    conn.close()

    data = list()
    for (circuit_str, fidelity_val, params_str) in rows:
        try:
            # Convert parameters string to NumPy array.
            init_params = np.fromstring(params_str.strip("[]"), sep=" ")
            fidelity = fidelity_val

            if model == 'GCN':
                processed_circuit = process_graph(circuit_str, init_params=init_params, fidelity=fidelity,
                                                  gate_set=gate_set)
                data.append(processed_circuit)
            elif model == 'RandomForest':
                processed_circuit = circuit_to_tensor(circuit_str, fidelity, gate_set)
                data.append(processed_circuit)
            else:
                raise ValueError(f"Unknown model: {model}")
        except Exception as e:
            print(f"Error processing circuit {circuit_str}: {e}")

    return data


def save_data(path, data=None):
    """
    Save processed data to a file using PyTorch's save function.

    Args:
        path (str): File path where the data will be saved.
        data (object, optional): The data object to save.

    Returns:
        None
    """
    try:
        torch.save(data, path)
        print(f"Successfully saved to {path}: {len(data)} items")
    except Exception as e:
        print(f"An error occurred while saving the data: {str(e)}")


def load_data(path):
    """
    Load processed data from a file using PyTorch's load function.

    Args:
        path (str): File path from which to load the data.

    Returns:
        object: The loaded data object, or None if loading fails.
    """
    try:
        data = torch.load(path)
        print(f"Data successfully loaded from {path}")
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        return None