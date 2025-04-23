# Copyright 2025 Fraunhofer Institute for Open Communication Systems FOKUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

from config import DeviceConfig, PathConfig, get_model_config_from_path
from evaluate.evaluate_utils import load_rf_model, evaluate_rf, load_gnn_model, evaluate_gnn, analyze_circuits, \
    analyze_error_by_fidelity, analyze_gate_features, compute_metrics
from util.config_utils import get_gate_set_and_features_by_name
from util.data_loader import load_data


def run_random_forest_evaluation(gate_set, device, model_name, color, path_config):
    print("=== Random Forest Evaluation ===")
    model_name_rf = f"random_forest_{gate_set}"
    model_config_path = os.path.join(path_config.paths['trained_models'], f'{model_name_rf}_config.json')
    model_config_rf = get_model_config_from_path(model_config_path, model_name_rf)
    model_config_rf['device'] = device

    data_path_rf = os.path.join(path_config.paths['rf_data'], f'{model_name}.pt')
    dataset_rf = load_data(data_path_rf)
    print(f"Loaded {len(dataset_rf)} circuits for RF evaluation.")
    if len(dataset_rf) == 0:
        raise ValueError("No data available for RF evaluation.")

    rf_model = load_rf_model(f'{gate_set}')
    predicted_rf, actual_rf = evaluate_rf(dataset_rf, rf_model)
    tolerance_rf = getattr(model_config_rf, "accuracy_tolerance", 0.1)
    metrics_rf = compute_metrics(predicted_rf, actual_rf, label="Random Forest", tolerance=tolerance_rf)
    errors_rf = predicted_rf - actual_rf

    df_rf = pd.DataFrame({
        "Actual Fidelity": actual_rf,
        "Predicted Fidelity": predicted_rf,
        "Error": errors_rf
    })

    # ---------

    sns.set_theme(style="darkgrid")
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))

    # Scatter plot: Predicted vs. Actual
    sns.scatterplot(
        data=df_rf,
        x="Actual Fidelity",
        y="Predicted Fidelity",
        ax=axs[0],
        color=color,
        alpha=0.6,
        s=50
    )
    min_val_gnn = min(df_rf["Actual Fidelity"].min(), df_rf["Predicted Fidelity"].min())
    max_val_gnn = max(df_rf["Actual Fidelity"].max(), df_rf["Predicted Fidelity"].max())
    axs[0].plot([min_val_gnn, max_val_gnn], [min_val_gnn, max_val_gnn], 'r--', label="y = x")
    axs[0].set_title("Predicted vs. Actual Fidelity", fontsize=16)
    axs[0].set_xlabel("Actual Fidelity", fontsize=14)
    axs[0].set_ylabel("Predicted Fidelity", fontsize=14)
    axs[0].legend()

    # Error Distribution Histogram
    sns.histplot(
        data=df_rf,
        x="Error",
        bins=30,
        color=color,
        ax=axs[1]
    )
    axs[1].set_title("Error Distribution (Predicted - Actual)", fontsize=16)
    axs[1].set_xlabel("Error", fontsize=14)
    axs[1].set_ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ---------

    metrics_save_path_rf = os.path.join(path_config.paths['benchmark'], "rf_evaluation_metrics.csv")
    df_rf.to_csv(metrics_save_path_rf, index=False)
    print(f"RF metrics saved to {metrics_save_path_rf}\n")
    return metrics_rf


def run_gcn_evaluation(device, model_name, data_set_name, gate_mapping, color, path_config):
    print("=== GNN Evaluation ===")
    print(model_name)
    model_config_path = os.path.join(path_config.paths['trained_models'], f'{model_name}', f'{model_name}_config.json')
    model_config = get_model_config_from_path(model_config_path, device)

    dataset_path = os.path.join(path_config.paths['gcn_data'], f'{data_set_name}.pt')
    circuits = load_data(dataset_path)
    print(f"Loaded {len(circuits)} circuits for GNN evaluation.")

    if not circuits:
        raise ValueError("No data available for GNN evaluation.")

    model_path = os.path.join(path_config.paths['trained_models'], f"{model_name}", f"{model_name}.pth")
    gnn_model = load_gnn_model(model_path, model_config)

    predicted, actual = evaluate_gnn(circuits, gnn_model, model_config)
    metrics = compute_metrics(predicted, actual, label="GNN")
    errors = predicted - actual

    output_dir = os.path.join(path_config.paths['trained_models'], f"{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Analysis ---
    run_circuit_analysis(circuits, predicted, actual, gate_mapping, color, output_dir, model_name)
    plot_gate_feature_stats(circuits, predicted, actual, gate_mapping, output_dir, model_name)
    plot_fidelity_error_stats(actual, predicted, output_dir)

    # --- Plotting ---
    df = pd.DataFrame({
        "Actual Fidelity": actual,
        "Predicted Fidelity": predicted,
        "Error": errors
    })

    plot_error_distribution(df, model_name, color, output_dir)
    plot_hex_density(df, model_name, color, output_dir)

    return metrics


def run_circuit_analysis(circuits, predicted, actual, gate_mapping, color, output_dir, model_name):
    analyze_circuits(
        circuits=circuits,
        predicted=predicted,
        actual=actual,
        gate_mapping=gate_mapping,
        top_percent=5.0,
        min_fidelity=None,
        output_dir=output_dir,
        file_prefix="problem_circuits_all",
        color=color,
        model_name=model_name,
    )
    analyze_circuits(
        circuits=circuits,
        predicted=predicted,
        actual=actual,
        gate_mapping=gate_mapping,
        top_percent=5.0,
        min_fidelity=0.9,
        output_dir=output_dir,
        file_prefix="problem_circuits_hifid",
        model_name=model_name
    )


def plot_gate_feature_stats(circuits, predicted, actual, gate_mapping, output_dir, model_name):
    stats = analyze_gate_features(
        circuits=circuits,
        predicted_scores=predicted,
        actual_scores=actual,
        gate_mapping=gate_mapping,
        model_name=model_name,
        output_dir=output_dir
    )
    print("\nGate Feature Stats (presence vs. error):")
    print(stats.head(10))


def plot_fidelity_error_stats(actual, predicted, output_dir):
    df_error = analyze_error_by_fidelity(
        actual_scores=actual,
        predicted_scores=predicted,
        num_bins=10,
        output_dir=output_dir
    )
    print("\nAverage error by fidelity bin:")
    print(df_error)


def plot_error_distribution(df, model_name, color, output_dir):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(18, 5))

    sns.histplot(data=df, x="Error", bins=30, color=color, ax=ax)
    ax.set_title("Error Distribution (Predicted - Actual)", fontsize=16)
    ax.set_xlabel("Error", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x / 1000)}T"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_error.png"))
    plt.show()


def plot_hex_density(df, model_name, color, output_dir):
    sns.set_theme(style="darkgrid")
    hex_plot = sns.jointplot(
        data=df,
        x="Actual Fidelity",
        y="Predicted Fidelity",
        kind="hex",
        color=color,
        height=8,
        ratio=7,
        joint_kws={
            "gridsize": 100,
            "mincnt": 1,
            "norm": mcolors.LogNorm()
        }
    )
    hex_plot.ax_joint.set_xlim(-0.01, 1.01)
    hex_plot.ax_joint.set_ylim(-0.01, 1.01)

    hex_plot.set_axis_labels("Actual Fidelity", "Predicted Fidelity", fontsize=14)
    hex_plot.fig.suptitle("Fidelity Density (Log Scale)", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_distr.png"))
    plt.show()


if __name__ == "__main__":
    # --- Common Setup ---
    gate_set = 'gate_set_ghz_a'
    data_set_name = 'test_ghz_a_squash'  # f'{gate_set}'
    model_name = 'demo_gcn_ghz_a_2025-04-19_15-14-17'  # f'{gate_set}'
    color = 'skyblue'
    gnn = True
    rf = False
    print(f"=== Evaluation on {data_set_name} ===")
    device_config = DeviceConfig()
    device = device_config.device
    path_config = PathConfig()
    gate_mapping, _ = get_gate_set_and_features_by_name(gate_set)

    if rf:
        metrics_rf = run_random_forest_evaluation(gate_set=gate_set, device=device, model_name=model_name,
                                                  path_config=path_config)
    if gnn:
        metrics_gnn = run_gcn_evaluation(device=device, model_name=model_name, data_set_name=data_set_name,
                                         gate_mapping=gate_mapping, color=color,
                                         path_config=path_config)
