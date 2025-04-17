import torch
import pickle
import torch_geometric.utils as pyg_utils
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch_geometric.data import Data
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import ControlledGate
from qiskit.qasm3 import loads

from config import QCConfig
from util.proxy import zero_proxy_GHZ


def qiskit_to_data_object(circuit, init_params=None, gate_set=None, max_params=1, num_qubits=3, proxy=False):
    """
    Convert a Qiskit circuit (in QASM 3.0 format) to a PyTorch Geometric Data object.

    This function performs the following steps:
      1. Loads the circuit from an OPENQASM 3.0 string using qiskit.qasm3.loads.
      2. Converts the circuit to a DAG representation.
      3. Optionally computes a proxy value (using zero_proxy_GHZ) if proxy=True.
      4. Converts the DAG to a NetworkX graph.
      5. Creates a PyG Data object from the NetworkX graph including node features, edge index,
         edge attributes, and an optional fitness label.

    Args:
        circuit (str): OPENQASM 3.0 string representing the quantum circuit.
        init_params (list, optional): Initial parameters for nodes (if available).
        gate_set (list): List of allowed gate names. Must be provided.
        max_params (int, optional): Maximum number of parameters per gate (default is 1).
        num_qubits (int, optional): Number of qubits in the circuit (default is 3).
        proxy (bool, optional): Flag to compute a proxy value using zero_proxy_GHZ (default False).

    Returns:
        Data: PyTorch Geometric Data object representing the circuit graph.

    Raises:
        Exception: If gate_set is None.
    """
    if gate_set is None:
        raise Exception("Error: The gate_set parameter cannot be None. Please provide a valid gate set.")

    qiskit_circuit = loads(circuit)
    dag_circuit = circuit_to_dag(qiskit_circuit)

    # Compute proxy value if requested.
    if proxy is True:
        proxy = zero_proxy_GHZ(qiskit_circuit, dag_circuit)
    else:
        proxy = None

    graph = convert_dag_to_networkx(dag_circuit)
    tg_qc = create_data_object(graph, init_params=init_params, gate_set=gate_set,
                               max_params=max_params, num_qubits=num_qubits, proxy=proxy)
    return tg_qc


def process_graph(qasm3_circuit, init_params=None, fidelity=None, gate_set=None, max_params=1, num_qubits=3,
                  proxy=False):
    """
    Process a QASM 3.0 circuit string to create a PyTorch Geometric Data object.

    This function is similar to qiskit_to_data_object but also embeds a fidelity value.
    It:
      1. Loads the circuit from a QASM 3.0 string.
      2. Converts it to a DAG representation.
      3. Optionally computes a proxy value.
      4. Converts the DAG to a NetworkX graph.
      5. Creates a PyG Data object containing fidelity (y) along with node and edge data.

    Args:
        qasm3_circuit (str): OPENQASM 3.0 circuit string.
        init_params (list, optional): Initial parameters for node processing.
        fidelity (float, optional): Fitness value to be attached to the Data object.
        gate_set (list): List of allowed gate names. Must be provided.
        max_params (int, optional): Maximum allowed parameters per gate (default 1).
        num_qubits (int, optional): Number of qubits in the circuit (default 3).
        proxy (bool, optional): Whether to compute a proxy value using zero_proxy_GHZ (default False).

    Returns:
        Data: PyTorch Geometric Data object with circuit representation and fitness value.

    Raises:
        Exception: If gate_set is not provided.
    """
    if gate_set is None:
        raise Exception("Error: The gate_set parameter cannot be None. Please provide a valid gate set.")

    qiskit_circuit = loads(qasm3_circuit)
    dag_circuit = circuit_to_dag(qiskit_circuit)

    if proxy is True:
        proxy = zero_proxy_GHZ(qiskit_circuit, dag_circuit)
    else:
        proxy = None

    graph = convert_dag_to_networkx(dag_circuit)
    data = create_data_object(graph, init_params=init_params, fidelity=fidelity,
                              gate_set=gate_set, max_params=max_params, num_qubits=num_qubits, proxy=proxy)
    return data


def convert_dag_to_networkx(dag_circuit: DAGCircuit):
    """
    Convert a Qiskit DAGCircuit into a NetworkX directed graph.

    Each qubit is added as a node, and each operation (gate) in topological order is added as
    a separate node with attributes (name, parameters, qubit arguments, controls, targets).
    Directed edges connect nodes to preserve the circuit's temporal ordering.

    Args:
        dag_circuit (DAGCircuit): DAG representation of a quantum circuit.

    Returns:
        tuple: (G, gate_types, qubit_indices) where:
            - G (nx.DiGraph): The constructed directed graph.
            - gate_types (set): Set of all gate names encountered.
            - qubit_indices (list): List of qubit indices from the original circuit.
    """
    G = nx.DiGraph()
    gate_types = set()
    qubit_indices = [q._index for q in dag_circuit.qubits]
    for qidx in qubit_indices:
        G.add_node(qidx)

    index = len(qubit_indices)
    last_node_on_qubit = {}

    # Process each operation node in topological order.
    for dag_node in dag_circuit.topological_op_nodes():
        node_id = index
        index += 1

        gate_name = dag_node.name
        gate_params = dag_node.op.params
        qargs = [q._index for q in dag_node.qargs]

        # If the gate is a ControlledGate, separate control and target qubits.
        if isinstance(dag_node.op, ControlledGate):
            num_ctrl_qubits = dag_node.op.num_ctrl_qubits
            control_qubits = [q._index for q in dag_node.qargs[:num_ctrl_qubits]]
            target_qubits = [q._index for q in dag_node.qargs[num_ctrl_qubits:]]
        else:
            control_qubits = []
            target_qubits = qargs

        G.add_node(
            node_id,
            name=gate_name,
            params=gate_params,
            qargs=qargs,
            controls=control_qubits,
            targets=target_qubits
        )
        gate_types.add(gate_name)

        # Connect the current operation node to the last node on each qubit.
        for qidx in qargs:
            prev_node = last_node_on_qubit.get(qidx, qidx)
            G.add_edge(prev_node, node_id)
            last_node_on_qubit[qidx] = node_id

    return (G, gate_types, qubit_indices)


def create_data_object(graph, init_params=None, fidelity=None, gate_set=None, max_params=1, num_qubits=3, proxy=None):
    """
    Create a PyTorch Geometric Data object from a NetworkX graph.

    This function prepares node features from gate names and parameters, builds edge indices and
    edge attributes based on qubit connectivity, and attaches a fitness value (y) and optional proxy data.

    Args:
        graph (tuple): A tuple (G, gate_types, qubit_indices) from convert_dag_to_networkx.
        init_params (list, optional): List of initial parameters to assign to gate nodes.
        fidelity (float, optional): Fitness label to attach to the graph.
        gate_set (list): List of allowed gate names. Must be provided.
        max_params (int, optional): Maximum number of parameters per gate (default 1).
        num_qubits (int, optional): Number of qubits in the circuit (default 3).
        proxy (float, optional): Proxy fitness value (if computed).

    Returns:
        Data: A PyTorch Geometric Data object containing:
            - x: Node features tensor.
            - edge_index: Edge index tensor.
            - edge_attr: Edge attribute tensor.
            - y (optional): Fitness value tensor.
            - proxy (optional): Proxy value tensor.
            - Additional attributes: num_gate_types and qubit_indices.

    Raises:
        Exception: If gate_set is not provided.
        ValueError: For missing or inconsistent gate information.
    """
    if gate_set is None:
        raise Exception("Error: The gate_set parameter cannot be None. Please provide a valid gate set.")

    G, gate_types, qubit_indices = graph

    # If initial parameters are provided, assign them to gate nodes
    if init_params is not None:
        idx = 0
        for node in sorted(G.nodes()):
            if node not in qubit_indices:
                node_data = G.nodes[node]
                if 'params' in node_data and node_data['params']:
                    if idx < len(init_params):
                        node_data['params'] = [init_params[idx]]
                        idx += 1
                    else:
                        node_data['params'] = [0.0]

    # Map original qubit indices to consecutive node IDs.
    qubit_index_to_id = {qidx: idx for idx, qidx in enumerate(sorted(qubit_indices))}
    num_gate_types = len(gate_set)

    # Validate and process each gate node.
    for node in sorted(G.nodes()):
        if node not in qubit_indices:
            node_data = G.nodes[node]
            gate_name = node_data.get('name', None)
            if gate_name is None:
                raise ValueError(f"Node {node} has no gate name attribute.")
            gate_params = node_data.get('params', [])
            # If the gate is not allowed, attempt special handling for the 'u' gate.
            if gate_name not in gate_set:
                if gate_name == 'u':
                    node_data['params'] = [gate_params[0]]
                    gate_params = node_data['params']
                else:
                    print(gate_name)
                    raise ValueError(
                        f"Gate '{gate_name}' encountered in node {node} is not in the allowed gate set: {gate_set}"
                    )
            if len(gate_params) > max_params:
                raise ValueError(
                    f"Gate '{gate_name}' in node {node} has {len(gate_params)} parameters, which exceeds the maximum allowed ({max_params})."
                )

    node_features = []
    node_ids = []

    # Build node feature tensor.
    for node_id in sorted(G.nodes()):
        node_ids.append(node_id)
        if node_id in qubit_indices:
            # For qubits, set features to zero.
            node_feature = torch.zeros(num_gate_types + max_params)
        else:
            node_data = G.nodes[node_id]
            gate_name = node_data['name']
            if gate_name == 'u':
                gate_name = 'id'
            gate_type_id = gate_set.index(gate_name)  # Lookup the gate's index.
            gate_type_one_hot = F.one_hot(torch.tensor(gate_type_id), num_classes=num_gate_types).float()

            params = node_data['params']
            params_tensor = torch.zeros(max_params, dtype=torch.float)
            for i, param in enumerate(params):
                if i < max_params:
                    params_tensor[i] = param
            # Concatenate one-hot encoding of gate type with its parameters.
            node_feature = torch.cat([gate_type_one_hot, params_tensor])
        node_features.append(node_feature)

    try:
        x = torch.stack(node_features)
    except RuntimeError as e:
        raise RuntimeError("Error stacking node features. Check that all nodes have consistent feature sizes.") from e

    # Build edge list and corresponding attributes.
    edge_list = []
    edge_attrs = []
    for u, v in G.edges():
        edge_list.append((u, v))
        v_data = G.nodes[v]
        qubit_attr = torch.zeros(num_qubits)
        # Set attribute -1 for control qubits.
        for qidx in v_data.get('controls', []):
            if qidx not in qubit_index_to_id:
                raise ValueError(
                    f"Control qubit index {qidx} in node {v} is not among the initial qubit indices: {qubit_indices}")
            idx = qubit_index_to_id[qidx]
            if idx < num_qubits:
                qubit_attr[idx] = -1.0
        # Set attribute 1 for target qubits.
        for qidx in v_data.get('targets', []):
            if qidx not in qubit_index_to_id:
                raise ValueError(
                    f"Target qubit index {qidx} in node {v} is not among the initial qubit indices: {qubit_indices}")
            idx = qubit_index_to_id[qidx]
            if idx < num_qubits:
                qubit_attr[idx] = 1.0
        edge_attrs.append(qubit_attr)

    try:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    except Exception as e:
        raise ValueError("Error converting edge list to tensor. Check edge list formatting.") from e

    edge_attr = torch.stack(edge_attrs) if edge_attrs else None

    # Attach the fitness label (y) and optional proxy information.
    if fidelity is not None:
        if not isinstance(fidelity, (float, int)):
            raise TypeError("Fidelity must be a float or int value.")
        y = torch.tensor([fidelity], dtype=torch.float)
        if proxy is not None:
            proxy = torch.tensor([proxy], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, proxy=proxy)
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        if proxy is not None:
            proxy = torch.tensor([proxy], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, proxy=proxy)
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Attach additional attributes for later use.
    data.num_gate_types = num_gate_types
    data.qubit_indices = qubit_indices

    return data


def visualize_graph(data, gate_set=None):
    """
    Visualize a PyTorch Geometric Data object's graph.

    Converts the Data object into a NetworkX graph and then draws the graph using
    a spring layout. Node labels are assigned based on gate types or qubit indices.

    Args:
        data (Data): PyG Data object containing the graph representation.
        gate_set (list): List of allowed gate names. Must be provided.

    Raises:
        Exception: If gate_set is not provided.
        ValueError: If a node's gate ID is unknown.
    """
    if gate_set is None:
        raise Exception("Error: The gate_set parameter cannot be None. Please provide a valid gate set.")

    # Convert Data to a NetworkX graph.
    G = pyg_utils.to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=False)

    # Create labels for nodes: qubit nodes are labeled 'qu_<index>', others by gate type.
    gate_type_labels = {i: torch.argmax(feat[:data.num_gate_types]).item() for i, feat in enumerate(data.x)}
    gate_type_to_name = {idx: gate for idx, gate in enumerate(gate_set)}
    labels = {}
    for i in G.nodes():
        if i in data.qubit_indices:
            labels[i] = f'qu_{i}'
        else:
            gate_label = gate_type_labels.get(i, None)
            if gate_label is None or gate_label not in gate_type_to_name:
                raise ValueError(
                    f"Unknown gate ID {gate_label} encountered in node {i}. Allowed gate IDs: {list(gate_type_to_name.keys())}"
                )
            labels[i] = gate_type_to_name[gate_label]

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightgrey',
            edge_color='gray', node_size=500)
    plt.title("Quantum Circuit Graph Representation")
    plt.show()


if __name__ == '__main__':

    # Example circuit in OPENQASM 3.0 format.
    circuit = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    rx(10) q[0];
    ry(19) q[2];
    h q[0];
    h q[1];
    cx q[0], q[1];
    cx q[1], q[2];
    h q[2];
    """

    # Load QC configuration containing the allowed gate set.
    qc_config = QCConfig()

    # Process the circuit to create a Data object.
    data = process_graph(circuit, fidelity=1, gate_set=qc_config.gate_set_1)

    # Print circuit and converted data attributes.
    print(circuit)
    print("Node features (x):")
    print(data.x)
    print("\nEdge index:")
    print(data.edge_index)
    print("\nEdge attributes (edge_attr):")
    print(data.edge_attr)
    print("\nFitness value (y):")
    print(data.y)

    # Visualize the constructed graph.
    visualize_graph(data, gate_set=qc_config.gate_set_1)
