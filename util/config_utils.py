from config import QCConfig


def get_gate_set_and_features_by_name(gate_set_name):
    q_config = QCConfig()
    if gate_set_name == 'gate_set_ghz_a':
        gs = q_config.gate_set_ghz_a
        f = q_config.features_ghz_a
    elif gate_set_name == 'gate_set_ghz_b':
        gs = q_config.gate_set_ghz_b
        f = q_config.features_ghz_b
    elif gate_set_name == 'gate_set_ls_a':
        gs = q_config.gate_set_ls_a
        f = q_config.features_ls_a
    else:
        raise ValueError(f"Unknown gate set name: {gate_set_name}")
    return gs, f