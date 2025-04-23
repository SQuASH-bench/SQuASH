import numpy as np

ghz_a_conf = {
    "num_qubits": 3,
    "gate_set": ['cx', 'h', 'rx', 'ry', 'rz', 'id'],
    "layer_set": [],
    "max_depth": 6,
    "optimizer": "COBYLA",
    "optimizer_maxiter": 300,
    "optimizer_bound_min": -np.pi,
    "optimizer_bound_max": np.pi,
    "initial_param_min": -0.01,
    "initial_param_max": 0.01,

}
