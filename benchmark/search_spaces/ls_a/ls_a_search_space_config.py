ls_a_conf = {
    "num_qubits": 4,
    "gate_set": ['cx', 'h', 'rx', 'ry', 'swap', 'crx', 'cry'],
    "layer_set": ['RyRotationLayer', 'RxRotationLayer'],
    "max_depth": 10,
    "optimizer": "COBYLA",
    "optimization_maxiter_per_circuit": 200,
    "initial_param_min": -0.01,
    "initial_param_max": 0.01,
    #data
    "seed_dataset_generation": 274,
    "n_samples": 300,
    "n_features": 8,
    "margin": 1,
}
