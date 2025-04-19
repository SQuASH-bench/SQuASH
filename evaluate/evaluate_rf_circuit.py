# Copyright 2025 Fraunhofer Institute for Open Communication Systems FOKUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pickle

from config import PathConfig, QCConfig
from util.rf_convert import circuit_to_tensor


def load_trained_model(model_name):
    """
    Load a pre-trained Random Forest model from a pickle file.
    """
    paths = PathConfig().paths
    model_path = os.path.join(paths['trained_models'], f'{model_name}.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_circuit(model, circuit, fidelity=None, gate_set=None):
    """
    Evaluate a quantum circuit using the loaded Random Forest model.

    Converts the provided circuit into a tensor representation using a helper function,
    preprocesses the resulting tensor, and then performs a prediction using the model.
    The predicted fitness score is compared with the provided fidelity.

    Parameters:
    -----------
    model : object
        The loaded Random Forest model that supports the 'predict' method.
    circuit : object
        The quantum circuit to be evaluated.
    fidelity : float
        The fidelity value for the circuit, used during conversion.
    gate_set : any
        The set of allowed or supported gates used during the conversion.

    Returns:
    --------
    float
        The predicted fitness value for the quantum circuit as computed by the model.

    """
    if gate_set is None:
        raise ValueError("Error: The gate_set parameter cannot be None.")

    # Convert the circuit into a tensor representation; the conversion function returns a tuple.
    data_qc = circuit_to_tensor(circuit, gate_set, fidelity)
    X, _ = data_qc

    if hasattr(X, "cpu"):
        X = X.cpu().numpy()

    X = X.reshape(1, -1)

    prediction_array = model.predict(X)

    return float(prediction_array[0])


if __name__ == "__main__":
    paths = PathConfig().paths

    gate_set = 'gs1'
    model_name = f'random_forest_{gate_set}'

    rf_model = load_trained_model(model_name)

    circuit_file = '../data/merged_data/gs1/0/qc_ga_5_1bd6bd8da6ccc2d0fe967784d8d2bbb9b406bc560727e050e29725d4d67e68c2.pkl'

    with open(circuit_file, 'rb') as file:
        loaded_data = pickle.load(file)
        circuit = loaded_data.get("initial_circuit")
        fidelity = loaded_data.get("fidelity")

    prediction = evaluate_circuit(rf_model, circuit, fidelity, QCConfig().gate_set_ghz_b)

    print(f"Predicted fitness for the circuit: {prediction}")
    print(f"Ground truth fitness: {fidelity}")
