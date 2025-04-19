from config import QCConfig


def get_gate_set_and_features_by_name(gate_set_name):
    q_config = QCConfig()
    if 'ghz_a' in gate_set_name:
        gs = q_config.gate_set_ghz_a
        f = q_config.features_ghz_a
    elif 'ghz_b' in gate_set_name:
        gs = q_config.gate_set_ghz_b
        f = q_config.features_ghz_b
    elif 'ls_a' in gate_set_name:
        gs = q_config.gate_set_ls_a
        f = q_config.features_ls_a
    else:
        raise ValueError(f"Unknown gate set name: {gate_set_name}")
    return gs, f