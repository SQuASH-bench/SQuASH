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

from qiskit import  QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives.backend_estimator_v2 import BackendEstimatorV2
from qiskit_aer import Aer
from qiskit_machine_learning.optimizers import COBYLA
from benchmark.search_spaces.ls_a.generate_data import generate_data
from benchmark.search_spaces.ls_a.ls_a_search_space_config import ls_a_conf
from benchmark.utils import load_gnn_benchmark_model, predict_circuit_performance, \
    convert_qasm_circuit_into_trainable_pqc
from qiskit.qasm3 import loads
from util.config_utils import get_gate_set_and_features_by_name
from qiskit_machine_learning.algorithms import VQC


def standard_rx_ry_feature_map(n_features):
    assert n_features % 2 == 0, "Number of features must be even (2 features per qubit)."

    num_qubits = n_features // 2
    qc = QuantumCircuit(num_qubits)

    # Create parameter vector
    x_params = ParameterVector('x', n_features)

    for qubit in range(num_qubits):
        qc.rx(x_params[2 * qubit], qubit)
        qc.ry(x_params[2 * qubit + 1], qubit)
    return qc


def minimize_circ(circuit, params, search_space_config, X_train, y_train, X_test, y_test):
    print("Start calculation ground truth performance")
    feature_map = standard_rx_ry_feature_map(search_space_config["n_features"])
    backend = Aer.get_backend('statevector_simulator')
    initial_point = params
    estimator = BackendEstimatorV2(backend=backend)
    ansatz = circuit
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=search_space_config["optimization_maxiter_per_circuit"]),
        initial_point=initial_point
    )
    vqc.neural_network.estimator = estimator
    vqc.fit(X_train, y_train)
    train_score = vqc.score(X_train, y_train)
    test_score = vqc.score(X_test, y_test)

    if vqc.weights.__sizeof__() > 0:
        initial_ansatz = vqc.ansatz.copy()
        param_dict = dict(zip(initial_ansatz.parameters, vqc.initial_point))
        initial_bound_ansatz = initial_ansatz.assign_parameters(param_dict)
        optimal_ansatz = vqc.ansatz.copy()
        param_dict = dict(zip(optimal_ansatz.parameters, vqc.weights))
        optimal_bound_ansatz = optimal_ansatz.assign_parameters(param_dict)
    else:
        initial_bound_ansatz = vqc.ansatz
        optimal_bound_ansatz = vqc.ansatz
    return train_score, optimal_bound_ansatz, initial_bound_ansatz


if __name__ == "__main__":
    # specify search space
    search_space = "ls_a"
    search_space_config = ls_a_conf

    # specify model
    model_name = "gcn_ls_a"
    model = load_gnn_benchmark_model(model_name, search_space=search_space, device="cpu")

    # specify gate set and features
    gate_set, num_features = get_gate_set_and_features_by_name(search_space)

    # input circuit
    qasm_str = """
        OPENQASM 3.0;
    include "stdgates.inc";
    qubit[4] q;
    cx q[1], q[0];
    rx(-0.0001882318397238448) q[3];
    cx q[3], q[2];
    cry(0.0014645677474361494) q[0], q[1];
    rx(-0.003169948762661097) q[3];
    rx(-0.00801270768507796) q[1];
    rx(0.0026807015114686996) q[2];
    rx(0.009542108897590442) q[0];
    ry(-0.007253983281321652) q[1];
    ry(-0.0010626881216368404) q[2];
    ry(0.0033030770594475005) q[3];
    ry(0.001238955218084455) q[0];
    cry(-0.0009151219092275349) q[1], q[3];
    crx(-0.006164457201349616) q[0], q[2];
    ry(0.007700178811306678) q[0];
    crx(0.004316282921345862) q[2], q[0];
    cry(0.002608090273323144) q[3], q[0];
    ry(0.00637818988793345) q[0];
    ry(0.0007732452526005226) q[1];
    ry(0.009523718284082298) q[3];
    ry(0.00922988766313885) q[2];
    cry(0.007542081035513628) q[1], q[2];
    """
    X_train, y_train, X_test, y_test = generate_data()
    qc = loads(qasm_str)

    prediction = predict_circuit_performance(qasm_str, model, gate_set)
    print(f"Predicted performance for the trained PQC: {prediction}")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    trainable_circuit, params = convert_qasm_circuit_into_trainable_pqc(qasm_str, search_space_config["num_qubits"])
    ground_truth_fidelity_after_training, optimized_circuit, initial_circuit = minimize_circ(trainable_circuit, params,
                                                                                             search_space_config,
                                                                                             X_train, y_train,
                                                                                             X_test, y_test)
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print(f"Ground truth performance for the trained PQC: {ground_truth_fidelity_after_training}")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("Ground truth optimized circuit", {optimized_circuit.draw()})
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("Initial circuit", {initial_circuit.draw()})
