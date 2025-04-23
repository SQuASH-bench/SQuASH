from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from benchmark.search_spaces.ghz_b.ghz_b_search_space_config import ghz_b_conf


def get_ghz_b_target(num_qubits=ghz_b_conf["num_qubits"]):
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.save_statevector()
    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled).result()
    return result.get_statevector()