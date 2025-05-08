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

from qiskit import transpile
from qiskit.quantum_info import state_fidelity
from qiskit_aer import Aer
from scipy.optimize import minimize

from benchmark.search_spaces.ghz_a.generate_target_ghz_a import get_ghz_a_target
from benchmark.search_spaces.ghz_a.ghz_a_search_space_config import ghz_a_conf
from benchmark.utils import load_gnn_benchmark_model, predict_circuit_performance, \
    convert_qasm_circuit_into_trainable_pqc
from qiskit.qasm3 import loads
from util.config_utils import get_gate_set_and_features_by_name


def get_performance(qc):
    backend = Aer.get_backend('statevector_simulator')
    transpiled_circuit = transpile(qc, backend)
    job = backend.run(transpiled_circuit)
    job_result = job.result()
    statevector = job_result.get_statevector()
    return state_fidelity(statevector, get_ghz_a_target())


def fidelity_loss(params):
    param_dict = dict(zip(trainable_circuit.parameters, params))
    if params.__sizeof__() > 0:
        bound_circuit = trainable_circuit.assign_parameters(param_dict)
    else:
        bound_circuit = trainable_circuit
    fidelity = get_performance(bound_circuit)
    return -fidelity


def minimize_circ(circuit, params, search_space_config):
    initial_params = params
    bounds = [(search_space_config["optimizer_bound_min"], search_space_config["optimizer_bound_max"])] * len(
        circuit.parameters) if circuit.parameters else None
    if circuit.parameters:
        result = minimize(fidelity_loss,
                          initial_params,
                          method=search_space_config["optimizer"],
                          options={'maxiter': search_space_config["optimizer_maxiter"]},
                          bounds=bounds)
        optimal_fidelity = result.fun
        optimal_params = result.x
        param_dict = dict(zip(circuit.parameters, optimal_params))
        optimal_circuit = circuit.assign_parameters(param_dict)
    else:
        optimal_fidelity = get_performance(circuit)
        optimal_circuit = circuit

    return optimal_fidelity, optimal_circuit


if __name__ == "__main__":
    # specify search space
    search_space = "ghz_b"
    search_space_config = ghz_a_conf

    # specify model
    model_name = "gcn_proxy_augmented_ghz_b"
    proxy = True
    model = load_gnn_benchmark_model(model_name, search_space=search_space, device="cpu")

    # specify gate set and features
    gate_set, num_features = get_gate_set_and_features_by_name(search_space)

    # input circuit
    qasm_str = """
    OPENQASM 3.0;
    include "stdgates.inc";
    gate rzz(p0) _gate_q_0, _gate_q_1 {
      cx _gate_q_0, _gate_q_1;
      rz(p0) _gate_q_1;
      cx _gate_q_0, _gate_q_1;
    }
    qubit[3] q;
    sx q[0];
    x q[2];
    cz q[0], q[1];
    x q[2];
    sx q[2];
    sx q[1];
    rzz(0.008004549097265957) q[0], q[1];
    cz q[0], q[2];
    cz q[1], q[0];
    rz(-0.0001645752713190455) q[0];
    x q[2];
    sx q[2];
    rx(-0.0059563086747590815) q[0];
    id q[0];
    rx(-0.008005582434209091) q[1];
    rx(0.0008744532151137123) q[0];
    rz(-0.0011058669905589218) q[1];
    rz(-0.005602781028897132) q[2];
    """

    qc = loads(qasm_str)
    print("Initial circuit", loads(qasm_str))
    print("Performance of untrained input PQC", get_performance(qc))
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

    prediction = predict_circuit_performance(qasm_str, model, gate_set, proxy=proxy)
    print(f"Predicted performance for the trained PQC: {prediction}")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    trainable_circuit, params = convert_qasm_circuit_into_trainable_pqc(qasm_str, search_space_config["num_qubits"])
    ground_truth_fidelity_after_training, optimized_circuit = minimize_circ(trainable_circuit, params,
                                                                            search_space_config)
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print(f"Ground truth performance for the trained PQC: {-ground_truth_fidelity_after_training}")
    print("Ground truth optimized circuit", {optimized_circuit.draw()})
