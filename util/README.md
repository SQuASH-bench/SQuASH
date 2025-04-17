
# Utils

---

### Contents
1. [Data Loader](#data-loader)
2. [Quantum Circuit to Graph Conversion](#quantum-circuit-to-graph-conversion)
   - [Processing Quantum Circuit Files](#processing-quantum-circuit-files)
   - [Visualizing the Graph](#visualizing-the-graph)
3. [Dataset Creation](#dataset-creation)
4. [Metrics and Utility Functions](#metrics-and-utility-functions)

---

## Data Loader

The data loader functions enable the creation and management of datasets for quantum circuit regression tasks. It includes methods to process quantum circuit files, filter circuits based on specific criteria, and generate PyTorch Geometric Data objects.

---

## Quantum Circuit to Graph Conversion

### Processing Quantum Circuit Files

To convert a quantum circuit in QPY format into a graph, use the `process_graph` function. This function reads the circuit file, converts it into a `DAGCircuit` object, and processes it into a `NetworkX` graph. The resulting graph is converted into a PyTorch `Data` object containing the node features, edge indices, and edge attributes. Additionally, the circuit's fitness value is used as the target `(y)`.

#### Key Functions:

- **`process_graph(qiskit_circuit, fitness_value)`**:
   - Converts the Qiskit circuit into a graph and returns the PyTorch Geometric `Data` object.
   - Includes the fitness value, node features, and edge information.

- **`convert_dag_to_networkx(dag_circuit)`**:
   - Transforms the DAG circuit from Qiskit into a `NetworkX` directed graph, capturing gate types, qubits, and gate parameters.

- **`create_data_object(graph_tuple, fitness_value)`**:
   - Converts the NetworkX graph into a PyTorch Geometric `Data` object.
   - Includes node features (one-hot encoded gate types and parameters), edge indices (for qubit connections and gate operations), and edge attributes (control/target qubits).

```python
example for /data/ga4qco_data/ghz_3_qubits/seed_0/gen_0_ind_1

     ┌───┐     
q_0: ┤ Y ├──■──
     └───┘┌─┴─┐
q_1: ──■──┤ X ├
     ┌─┴─┐├───┤
q_2: ┤ X ├┤ Y ├
     └───┘└───┘

Node features (x):
tensor([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], --> Node 0: Qubit q_0 (initial state, no gate applied, so it's all zeros) 
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], --> Node 1: Qubit q_1 (initial state, no gate applied, all zeros) 
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], --> Node 2: Qubit q_2 (initial state, no gate applied, all zeros) 
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], --> Node 3: Y-gate applied to q_0 (one-hot vector indicating Y-gate type) 
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], --> Node 4: CX gate applied between q_0 and q_1 (one-hot encoding for X gate) 
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], --> Node 5: X-gate applied to q_1 (one-hot vector for X gate) 
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) --> Node 6: Y-gate applied to q_2 (one-hot vector for Y gate)

Edge index:
tensor([
        [0, 1, 2, 3, 4, 4], --> source node
        [3, 4, 4, 5, 5, 6]  --> target node
        ])
        
- [0, 3] --> Edge from qubit q_0 to the Y-gate applied on q_0.
- [1, 4] --> Edge from qubit q_1 to the CX gate (q_0 is control, q_1 is target).
- [2, 4] --> Edge from qubit q_2 to the CX gate (target qubit).
- [3, 5] --> Edge from the Y-gate applied to q_0 to the X-gate applied to q_1.
- [4, 5] --> Edge from the CX gate to the X-gate on q_1.
- [4, 6] --> Edge from the CX gate to the Y-gate applied on q_2.

Edge attributes (edge_attr):
tensor([
        [ 1., 0., 0.], --> Y-gate: q_0 is the target 
        [ 0., -1., 1.], --> CX gate: q_0 control, q_1 target 
        [ 0., -1., 1.], --> CX gate: q_0 control, q_2 target 
        [-1., 1., 0.], --> X-gate: q_1 is the target 
        [-1., 1., 0.], --> X-gate: q_1 is the target 
        [ 0., 0., 1.] --> Y-gate: q_2 is the target 
        ])
        
- First row `[1., 0., ...]` indicates a Y-gate on qubit q_0, with q_0 being the target (marked by `1`).
- Second row `[0., -1., 1., ...]` describes a CX gate with q_0 as the control (marked `-1`) and q_1 as the target (marked `1`).
- Similarly, other rows indicate different gate operations and their respective control/target qubits.

```

---

### Visualizing the Graph

The `visualize_graph` function converts the PyTorch `Data` object back into a NetworkX graph and plots it using Matplotlib. Node labels correspond to quantum gate types, and edges represent qubit interactions.

**Example**:

```python
visualize_graph(data)
```

**Output**: A graphical representation of the quantum circuit graph.

---

# Quantum Circuit to Tensor Conversion

## Overview

This utility converts an OPENQASM 3.0 quantum circuit into a fixed-size feature tensor for model training and evaluation. The conversion includes:

- **Gate Encoding:** One-hot encoding for each gate (with a reserved index for no-op).
- **Qubit Usage Mapping:** Encoding of qubit roles (e.g., control as -1, target as +1).
- **Parameter Extraction:** Collection and fixed-length padding of gate parameters.
- **Depth Normalization:** Padding of the feature matrix with no-op rows to reach a specified circuit depth.
- **Optional Fidelity Output:** Optionally returns a target fidelity value.

## How It Works

1. **Parsing the Circuit:**  
   The circuit is parsed using Qiskit's `loads` function from an OPENQASM 3.0 string.

2. **Feature Extraction per Instruction:**  
   For each gate instruction:
   - **Gate Name Encoding:**  
     The gate name is one-hot encoded based on a provided gate set (with an extra slot for a no-op).
   - **Qubit Usage:**  
     - Controlled gates require exactly 2 qubits, marking the control with -1 and the target with +1.
     - Single or two-qubit non-controlled gates mark active qubits with +1.
   - **Gate Parameters:**  
     Gate parameters are gathered, padded (or truncated) to a fixed length.
   - **Assemble Row:**  
     The one-hot encoded gate, qubit usage vector, and parameters are concatenated into a row.

3. **Matrix Padding:**  
   If the circuit contains fewer instructions than the maximum depth, additional no-op rows are appended to form a tensor of fixed dimensions.

4. **Output Generation:**  
   Returns a tensor `X` representing the circuit, and optionally a tensor `y` with the fidelity value.

---