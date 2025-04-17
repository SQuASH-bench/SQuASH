import sqlite3
import networkx as nx

from qiskit.qasm3 import loads
from qiskit.converters import circuit_to_dag


def load_initial_circuits():
    """
    Load initial quantum circuits and their corresponding fidelities from a SQLite database.

    The function connects to the "unique_circuits_gs1.db" database, retrieves the 'initial_circuit'
    and 'fidelity' columns from the "circuits" table, and returns the fetched tuples.

    Returns:
        list of tuples: Each tuple contains (initial_circuit, fidelity) for a circuit.
    """
    DB_PATH = "unique_circuits_gs1.db"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Retrieve all circuits with their fidelity values.
    cursor.execute("SELECT initial_circuit, fidelity FROM circuits")
    # Uncomment the following line to filter circuits by fidelity (e.g., fidelity >= 0.99)
    # cursor.execute("SELECT initial_circuit, fidelity FROM circuits WHERE fidelity >= 0.99")

    results = cursor.fetchall()  # Results is a list of (circuit, fidelity) tuples.
    conn.close()
    return results


def check_superposition(circuit):
    """
    Determine whether the input circuit creates a superposition state.

    The function converts the quantum circuit to a DAG representation and checks if
    at least one qubit undergoes an H-like transformation (Hadamard, RX(π/2), or RY(π/2)).

    Args:
        circuit: A quantum circuit object.

    Returns:
        bool: True if a superposition gate is found, False otherwise.
    """
    # Convert the circuit to a Directed Acyclic Graph (DAG)
    dag = circuit_to_dag(circuit)
    # Define a set of gate names that can create a superposition.
    superposition_gates = {'h', 'rx', 'ry'}

    # Iterate over operation nodes in the DAG.
    for node in dag.op_nodes():
        if node.op.name in superposition_gates:
            return True
        # Check for RX(π/2) within a small tolerance
        if node.op.name == 'rx' and abs(float(node.op.params[0]) - 1.5708) < 0.01:
            return True
        # Check for RY(π/2) within a small tolerance
        if node.op.name == 'ry' and abs(float(node.op.params[0]) - 1.5708) < 0.01:
            return True
    return False


def build_entanglement_graph(dag, entangling_gates={'cx', 'cz', 'rzz'}):
    """
    Build a graph representing entanglement connectivity within a quantum circuit.

    Nodes in the graph represent qubits, and edges represent entangling operations defined
    by the provided set of gate names (defaults to 'cx', 'cz', and 'rzz').

    Args:
        dag: The DAG representation of a quantum circuit.
        entangling_gates (set, optional): A set of gate names that indicate entangling operations.

    Returns:
        networkx.Graph: An undirected graph of qubit connectivity via entangling operations.
    """
    G = nx.Graph()

    # Add all qubit indices as nodes
    qubit_indices = [q._index for q in dag.qubits]
    G.add_nodes_from(qubit_indices)

    # Iterate over the operation nodes to add edges between qubits entangled by specific gates.
    for node in dag.op_nodes():
        if node.op.name in entangling_gates:
            qubits_involved = [q._index for q in node.qargs]
            if len(qubits_involved) == 2:
                G.add_edge(qubits_involved[0], qubits_involved[1])
    return G


def zero_proxy_GHZ(circuit, dag):
    """
    Evaluate if a quantum circuit has the structural capability to generate a GHZ state.

    This is done by ensuring:
      - At least one qubit is put into superposition (using H, RX(π/2), or RY(π/2)).
      - The entanglement graph (constructed from the circuit's DAG) is fully connected,
        meaning it has enough entangling gates to connect all qubits.

    Args:
        circuit: The quantum circuit object.
        dag: The DAG representation of the circuit.

    Returns:
        float: 1.0 if the circuit passes the structural test for GHZ state generation,
               0.0 otherwise.
    """
    # Check if the circuit creates any superposition.
    has_superposition = check_superposition(circuit)

    # Build the entanglement graph and analyze its connectivity.
    G = build_entanglement_graph(dag)
    is_connected = nx.is_connected(G)
    # A minimally connected graph (a tree) should have number_of_nodes - 1 edges.
    min_edges = len(G.nodes) - 1
    sufficient_edges = G.number_of_edges() >= min_edges

    # Decision making based on superposition and connectivity conditions.
    if not has_superposition:
        return 0.0  # Circuit cannot generate GHZ without superposition.
    if not is_connected or not sufficient_edges:
        return 0.0  # Circuit is not sufficiently entangled.

    return 1.0  # Circuit has the structural elements to generate a GHZ state.

