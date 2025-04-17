import torch
from qiskit.qasm3 import loads
from config import QCConfig


def circuit_to_tensor(qasm3, fidelity=None, gate_set=["cx", "h", "rx", "ry", "rz", "id"], num_qubits=3, max_params=1,
                      depth=26):
    """
    Convert an OPENQASM 3.0 quantum circuit into a feature tensor suitable for machine learning models.

    The conversion process includes the following steps:
      1. One-hot encode the gate name.
      2. Map qubit usage for each gate:
         - For controlled gates: mark control with -1 and target with +1.
         - For single and two-qubit gates: mark each qubit as +1.
      3. Extract gate parameters and pad/truncate them to a fixed length.
      4. Assemble a feature row by concatenating the one-hot gate vector, qubit usage vector, and parameter vector.
      5. Pad the list of feature rows with "no-op" rows until the total number of rows equals the fixed circuit depth.

    Args:
        qasm3 (str): A string containing the OPENQASM 3.0 representation of the quantum circuit.
        fidelity (float, optional): A fidelity value for the circuit. If provided, the function returns a tuple (X, y);
                                    otherwise, it returns the feature tensor X.
        gate_set (list): A list of allowed gate names. The last index in the one-hot encoding is reserved for no-op.
        num_qubits (int): The total number of qubits in the circuit.
        max_params (int): Maximum number of parameters per gate to include.
        depth (int): Maximum circuit depth (number of instructions). If the circuit has fewer instructions,
                     the feature tensor is padded with no-op rows.

    Returns:
        X_torch (Tensor): A tensor of shape (depth, dim_gate_one_hot + num_qubits + max_params) representing the circuit.
        y_torch (Tensor, optional): A tensor containing the fidelity value if provided.

    Raises:
        ValueError: If an unsupported gate is encountered or if circuit depth exceeds the specified maximum.
    """
    # Load the circuit from the QASM 3.0 string.
    circuit = loads(qasm3)

    # The index reserved for no-op encoding.
    no_op_index = len(gate_set)

    # Dimensions for one-hot encoding and additional features.
    dim_gate_one_hot = len(gate_set) + 1  # Extra index for no-op.
    dim_qubits = num_qubits
    dim_params = max_params

    feature_rows = []

    # Process each instruction in the circuit.
    for instr in circuit.data:
        gate_name = instr.operation.name.lower()

        # ---------- 1) One-hot encode the gate name ----------
        gate_vec = [0] * dim_gate_one_hot
        if gate_name in gate_set:
            gate_idx = gate_set.index(gate_name)
            gate_vec[gate_idx] = 1
        elif gate_name == 'u':  # Treat U gate as identity ('id')
            gate_idx = gate_set.index("id")
            gate_vec[gate_idx] = 1
        else:
            raise ValueError(f"Unknown or unsupported gate: '{gate_name}'")

        # ---------- 2) Map qubit usage ----------
        # Initialize qubit usage vector with zeros.
        qubit_usage = [0] * dim_qubits
        qubits = instr.qubits

        # For controlled gates: require exactly two qubits.
        if gate_name.startswith('c'):
            if len(qubits) != 2:
                raise ValueError(
                    f"Controlled gate '{gate_name}' requires exactly 2 qubits, "
                    f"but this instruction has {len(qubits)} qubits."
                )
            control_index = qubits[0]._index
            target_index = qubits[1]._index
            if control_index >= num_qubits or target_index >= num_qubits:
                raise ValueError(
                    f"Qubit index out of range. Circuit has {num_qubits} qubits, "
                    f"but got control={control_index}, target={target_index}."
                )
            qubit_usage[control_index] = -1
            qubit_usage[target_index] = +1
        else:
            # For single-qubit gates.
            if len(qubits) == 1:
                q0 = qubits[0]._index
                if q0 >= num_qubits:
                    raise ValueError(
                        f"Qubit index {q0} is out of range for a {num_qubits}-qubit circuit."
                    )
                qubit_usage[q0] = +1
            # For gates acting on two qubits (non-controlled).
            elif len(qubits) == 2:
                q0 = qubits[0]._index
                q1 = qubits[1]._index
                if q0 >= num_qubits or q1 >= num_qubits:
                    raise ValueError(
                        f"Qubit indices {q0}, {q1} are out of range for a {num_qubits}-qubit circuit."
                    )
                qubit_usage[q0] = +1
                qubit_usage[q1] = +1
            else:
                raise ValueError(
                    f"Gate '{gate_name}' expected 1 or 2 qubits, but got {len(qubits)}."
                )

        # ---------- 3) Gather parameters ----------
        params = []
        if hasattr(instr.operation, "params"):
            for p in instr.operation.params:
                params.append(float(p))
        # Pad the parameter list to ensure it has exactly max_params elements.
        params = (params + [0.0] * dim_params)[:dim_params]

        # ---------- 4) Assemble the feature row ----------
        row = gate_vec + qubit_usage + params
        feature_rows.append(row)

        # Check that the circuit depth does not exceed the given maximum.
        if len(feature_rows) > depth:
            raise ValueError(f"[ERROR] Circuit depth exceeds the maximum {depth}.")

    # ---------- 5) Pad with no-op rows until the feature array reaches the desired depth ----------
    while len(feature_rows) < depth:
        no_op_vec = [0] * dim_gate_one_hot
        no_op_vec[no_op_index] = 1  # Set the no-op flag.
        qubit_usage = [0] * dim_qubits
        params = [0.0] * dim_params
        row = no_op_vec + qubit_usage + params
        feature_rows.append(row)

    # Convert the list of feature rows into a PyTorch tensor.
    X_torch = torch.tensor(feature_rows, dtype=torch.float32)
    # If a fidelity value is provided, create a corresponding tensor for y.
    if fidelity is not None:
        y_torch = torch.tensor(fidelity, dtype=torch.float32)
        return X_torch, y_torch
    else:
        return X_torch


if __name__ == "__main__":
    # Example OPENQASM 3.0 circuit.
    qasm_example = """OPENQASM 3.0;
    include "stdgates.inc";
    bit[3] c;
    qubit[3] q;
    h q[0];
    rx(1.57) q[1];
    ry(1.57) q[2];
    rz(0.5) q[0];
    cx q[0], q[1];
    id q[2];
    """

    # Load QC configuration to access the allowed gate set.
    qc_config = QCConfig()

    # Convert the circuit into a tensor feature matrix with a corresponding fidelity value.
    X, y = circuit_to_tensor(qasm_example, 0.5, qc_config.gate_set_ghz_a, depth=22)

    print("Feature array shape:", X.shape)
    print(X)
    print(y)
