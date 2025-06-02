import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns


def get_all_unparametrized_one_q_qates():
    return ["h", "id", "x", "y", "z", "sx"]


def get_all_parametrized_one_q_qates():
    return ["ry", "rx", "rz"]


def get_unparametrized_uncontrolled_two_q_gates():
    return ["swap"]


def get_unparametrized_controlled_two_q_gates():
    return ["cx", "cnot", "cz"]


def get_parametrized_uncontrolled_two_q_gates():
    return ["rzz"]


def get_parametrized_controlled_two_q_gates():
    return ["crx", "cry"]


def split_gate_set(gate_set):
    all_unparametrized_one_q_qates = get_all_unparametrized_one_q_qates()
    all_parametrized_one_q_qates = get_all_parametrized_one_q_qates()
    all_unparametrized_uncontrolled_two_q_gates = get_unparametrized_uncontrolled_two_q_gates()
    all_unparametrized_controlled_two_q_gates = get_unparametrized_controlled_two_q_gates()
    all_parametrized_uncontrolled_two_q_gates = get_parametrized_uncontrolled_two_q_gates()
    all_parametrized_controlled_two_q_gates = get_parametrized_controlled_two_q_gates()
    one_unparametrized_q_gates = []
    one_parametrized_q_gates = []
    unparametrized_uncontrolled_two_qubit_gates = []
    unparametrized_controlled_two_qubit_gates = []
    parametrized_uncontrolled_two_qubit_gates = []
    parametrized_controlled_two_qubit_gates = []

    for g in gate_set:
        if g in all_unparametrized_one_q_qates:
            one_unparametrized_q_gates.append(g)
        elif g in all_parametrized_one_q_qates:
            one_parametrized_q_gates.append(g)
        elif g in all_unparametrized_uncontrolled_two_q_gates:
            unparametrized_uncontrolled_two_qubit_gates.append(g)
        elif g in all_unparametrized_controlled_two_q_gates:
            unparametrized_controlled_two_qubit_gates.append(g)
        elif g in all_parametrized_uncontrolled_two_q_gates:
            parametrized_uncontrolled_two_qubit_gates.append(g)
        elif g in all_parametrized_controlled_two_q_gates:
            parametrized_controlled_two_qubit_gates.append(g)
        else:
            print(f"[ERROR] Gate {g} is not supported.")
            raise ValueError(f"Gate {g} is not supported.")
    return one_unparametrized_q_gates, one_parametrized_q_gates, unparametrized_uncontrolled_two_qubit_gates, unparametrized_controlled_two_qubit_gates, parametrized_uncontrolled_two_qubit_gates, parametrized_controlled_two_qubit_gates



def get_num_actions_from_gate_set_and_layers(gate_set, layer_set, num_qubits):
    one_unparam_q_gates, one_param_q_gates, unparametrized_uncontrolled_two_qubit_gates, unparametrized_controlled_two_qubit_gates, parametrized_uncontrolled_two_qubit_gates, parametrized_controlled_two_qubit_gates = split_gate_set(
        gate_set)
    num_actions = len(one_unparam_q_gates) * num_qubits + len(one_param_q_gates) *  num_qubits + len(unparametrized_uncontrolled_two_qubit_gates) * (
                          num_qubits * (num_qubits - 1) / 2) + len(parametrized_uncontrolled_two_qubit_gates) * (
                          num_qubits * (num_qubits - 1) / 2)  + len(unparametrized_controlled_two_qubit_gates) * (num_qubits * (num_qubits - 1)) + len(
        parametrized_controlled_two_qubit_gates) * (num_qubits * (num_qubits - 1)) + len(layer_set)
    #print("Num actions", int(num_actions))
    return int(num_actions)


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed(seed):
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_series(data):
    """
    Converts a list into a Pandas Series with integer index.
    """
    return pd.Series(data)

def moving_average(data, window_size):
    """Compute simple moving average with a given window size."""
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean()

def plot_results(loss_history_seed, rewards_history_seed, trained_only=True, smooth_window=50):
    """
    Plot training loss and total episode rewards for a single seed, with optional smoothing.

    Parameters:
        loss_history_seed: list of loss values over updates.
        rewards_history_seed: list or tuple of reward lists.
                              If not trained_only, expected format is (trained_rewards, random_rewards).
        trained_only: if True, only plots the trained agent’s rewards.
        smooth_window: integer, window size for moving average smoothing.
    """
    plt.figure(figsize=(10, 5))

    # Plot loss history
    plt.subplot(1, 2, 1)
    smoothed_loss = moving_average(loss_history_seed, smooth_window)
    sns.lineplot(data=smoothed_loss)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title(f'Loss over Time (window={smooth_window})')

    # Plot reward history
    plt.subplot(1, 2, 2)
    if trained_only:
        smoothed_trained_rewards = moving_average(rewards_history_seed, smooth_window)
        sns.lineplot(data=smoothed_trained_rewards, label="trained_agent", color='blue')
    else:
        smoothed_trained_rewards = moving_average(rewards_history_seed[0], smooth_window)
        smoothed_random_rewards = moving_average(rewards_history_seed[1], smooth_window)
        sns.lineplot(data=smoothed_trained_rewards, label="trained_agent", color='blue')
        sns.lineplot(data=smoothed_random_rewards, label="random_agent", color='red')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Total Reward per Episode (window={smooth_window})')
    plt.legend()
    plt.tight_layout()
    plt.show()



def get_exec_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return [hours, minutes, seconds]