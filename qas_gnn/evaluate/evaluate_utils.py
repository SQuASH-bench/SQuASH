import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr, spearmanr

from surrogate_rf import prepare_data
from surrogate_models.architectures.gnn.gnn import RegGNN
from config import PathConfig


# --------------------- Utility Functions --------------------- #

def compute_metrics(predicted, actual, label="All Data", tolerance=0.1):
    """
    Computes basic regression metrics:
      - Mean Absolute Error (MAE)
      - Mean Squared Error (MSE)
      - Root Mean Squared Error (RMSE)
      - R^2 (coefficient of determination)
      - Pearson correlation
      - Accuracy: Percentage of samples with |error| <= tolerance
    """
    errors = predicted - actual
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    mae = np.mean(abs_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else float('nan')

    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2) if len(actual) > 1 else 0
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')
    spearman, _ = spearmanr(actual, predicted)

    accuracy = 100.0 * np.mean(abs_errors <= tolerance)

    print(f"--- Metrics for {label} ---")
    print(f"Samples: {len(actual)}")
    print(f"MSE:     {mse:.4f}")
    print(f"MAE:     {mae:.4f}")
    print(f"RMSE:    {rmse:.4f}")
    print(f"R^2:     {r_squared:.4f}")
    print(f"Corr:    {correlation:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Accuracy (|err| <= {tolerance}): {accuracy:.2f}%\n")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "correlation": correlation,
        "r_squared": r_squared,
        "accuracy": accuracy
    }

# --------------------- Model Loading and Evaluation --------------------- #

def load_rf_model(model_name):
    """Loads a trained RandomForest model from a pickle file."""
    paths = PathConfig().paths
    model_path = os.path.join(paths['trained_models'], f"{model_name}.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_rf(dataset, model):
    """
    Evaluates a RandomForest model on a dataset.
    Assumes the model has a `prepare_data` method and an internal attribute `model`
    with a scikit-learn–style predict() method.
    """
    X, y = prepare_data(dataset)
    predicted = model.predict(X)
    return np.array(predicted), np.array(y)

def load_gnn_model(model_path, config):
    """Loads a trained GNN model from"""
    device = config.device
    model = RegGNN(
        num_layer=config.layer_num,
        emb_dim=config.emb_dim,
        edge_attr_dim=config.qubit_num,
        num_node_features=config.num_node_features,
        drop_ratio=config.drop_ratio,
        graph_pooling=config.graph_pooling,
        JK=config.JK,
        freeze_gnn=False,
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def evaluate_gnn(circuits, model, config):
    device = config.device
    loader = DataLoader(circuits, batch_size=config.batch_size, shuffle=False)

    predicted_scores = []
    actual_scores = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model.predict(data).cpu().numpy()
            act = data.y.cpu().numpy()
            predicted_scores.extend(pred)
            actual_scores.extend(act)

    return np.array(predicted_scores), np.array(actual_scores)

# --------------------- Gate Analysis Utilities (GNN only) --------------------- #

def get_gates(data, gate_mapping):
    """
    Given a circuit data object and a gate set (as a list),
    returns the list of gate names used in the circuit.
    Assumes data.x contains one-hot encoded gate features in its first N columns.
    """
    # Create a reverse mapping from index to gate name using enumerate
    reverse_map = {i: gate for i, gate in enumerate(gate_mapping)}
    num_gate_types = len(gate_mapping)
    gates = []
    x_np = data.x.cpu().numpy()  # shape: (num_nodes, num_gate_types + additional_features)
    for feat in x_np:
        one_hot = feat[:num_gate_types]
        if not np.allclose(one_hot, 0):
            gate_id = int(np.argmax(one_hot))
            gate_name = reverse_map.get(gate_id, f"UnknownGateID_{gate_id}")
            gates.append(gate_name)
    return gates

def plot_gate_usage_bar(gate_counts, title="Gate Usage in Problem Circuits", color=None, model_name=None):
    sns.set_theme(style="darkgrid")
    if not gate_counts:
        print("No gates to plot.")
        return
    df_usage = pd.DataFrame({
        "Gate": list(gate_counts.keys()),
        "Count": list(gate_counts.values())
    })
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_usage, x="Gate", y="Count", color=color)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Usage Count", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"../results/plots/gcn_{model_name}_gates.png")
    plt.show()


def analyze_circuits(circuits, predicted, actual, gate_mapping, top_percent,
                     min_fidelity=None, output_dir=None, file_prefix="problem_circuits", color=None, model_name=None):
    """
    Finds top X% error circuits. Optionally filters by min_fidelity.
    Plots and prints gate usage among these circuits. Saves CSV if output_dir provided.
    """
    print('[INFO] Start analyzing circuits')
    errors = np.abs(predicted - actual)

    if min_fidelity is not None:
        mask = (actual >= min_fidelity)
    else:
        mask = np.ones(len(actual), dtype=bool)

    filtered_indices = np.where(mask)[0]
    if len(filtered_indices) == 0:
        print(f"No circuits found with actual fidelity >= {min_fidelity}")
        return []

    filtered_errors = errors[filtered_indices]
    cutoff = np.percentile(filtered_errors, 100 - top_percent)
    problem_indices = filtered_indices[np.where(filtered_errors >= cutoff)]
    problem_indices_sorted = problem_indices[np.argsort(-errors[problem_indices])]

    print(f"Found {len(problem_indices_sorted)} circuits among the top {top_percent}% error", end='')
    if min_fidelity:
        print(f" (actual fidelity >= {min_fidelity}).")
    else:
        print(".")

    problem_details = []
    gate_counts = {}
    for idx in problem_indices_sorted:
        circuit_data = circuits[idx]
        circuit_gates = get_gates(circuit_data, gate_mapping)
        problem_details.append({
            "dataset_idx": idx,
            "actual_fidelity": float(actual[idx]),
            "predicted_fidelity": float(predicted[idx]),
            "abs_error": float(errors[idx]),
            "gates_used": circuit_gates
        })
        for g in circuit_gates:
            gate_counts[g] = gate_counts.get(g, 0) + 1

    if gate_counts:
        print("\nGate usage across circuits:")
        for gname, cnt in sorted(gate_counts.items(), key=lambda x: -x[1]):
            print(f"  {gname}: {cnt} occurrences")
        fidelity_str = f"(fidelity >= {min_fidelity})" if min_fidelity else "(All Fidelity)"
        plot_title = f"Gate Usage in Top {top_percent}% Error Circuits {fidelity_str}"
        plot_gate_usage_bar(gate_counts, title=plot_title, color=color, model_name=model_name)
    else:
        print("\nNo gates found among these circuits.")

    if output_dir and problem_details:
        os.makedirs(output_dir, exist_ok=True)
        details_path = os.path.join(output_dir, f"{file_prefix}_details.csv")
        df_details = pd.DataFrame(problem_details)
        df_details['gates_used'] = df_details['gates_used'].apply(lambda glist: ",".join(glist))
        df_details.to_csv(details_path, index=False)
        print(f"[INFO] Saved problem circuit details to: {details_path}")

        usage_path = os.path.join(output_dir, f"{file_prefix}_gate_usage.csv")
        usage_data = [{'gate': k, 'count': v} for k, v in sorted(gate_counts.items(), key=lambda x: -x[1])]
        pd.DataFrame(usage_data).to_csv(usage_path, index=False)
        print(f"[INFO] Saved gate usage summary to: {usage_path}")

    return problem_details

def analyze_gate_features(circuits, predicted_scores, actual_scores, gate_mapping, output_dir=None, color=None, model_name=None):
    errors = np.abs(predicted_scores - actual_scores)
    all_gates = set()
    for circuit in circuits:
        all_gates.update(get_gates(circuit, gate_mapping))
    all_gates = sorted(all_gates)

    data_list = []
    for i, circuit in enumerate(circuits):
        row_dict = {"abs_error": errors[i]}
        g_list = get_gates(circuit, gate_mapping)
        for gate in all_gates:
            row_dict[f"has_{gate}"] = int(gate in g_list)
        data_list.append(row_dict)
    df = pd.DataFrame(data_list)

    rows = []
    for gate in all_gates:
        col = f"has_{gate}"
        mask_present = (df[col] == 1)
        mask_absent = (df[col] == 0)
        if mask_present.sum() == 0:
            continue
        mean_err_present = df.loc[mask_present, "abs_error"].mean()
        mean_err_absent = df.loc[mask_absent, "abs_error"].mean() if mask_absent.sum() > 0 else np.nan
        diff_mean = mean_err_present - mean_err_absent
        corr_val, _ = pearsonr(df[col], df["abs_error"])
        rows.append({
            "gate": gate,
            "count_present": int(mask_present.sum()),
            "mean_err_present": mean_err_present,
            "mean_err_absent": mean_err_absent,
            "diff_mean_err": diff_mean,
            "corr_err_presence": corr_val
        })

    df_gate_stats = pd.DataFrame(rows).sort_values("diff_mean_err", ascending=False)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_gate_stats, x="gate", y="diff_mean_err", color=color)
    ax.set_title("Gate Presence vs. Error", fontsize=16)
    ax.set_ylabel("Difference in Mean Abs Error (Present - Absent)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig("../results/plots/gcn_{model_name}_gate.png")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, "gate_feature_stats.csv")
        df_gate_stats.to_csv(stats_path, index=False)
        print(f"[INFO] Saved gate feature stats to: {stats_path}")

    return df_gate_stats

def analyze_error_by_fidelity(actual_scores, predicted_scores, num_bins=10, output_dir=None):
    errors = np.abs(predicted_scores - actual_scores)
    bins = np.linspace(0, 1, num_bins + 1)
    indices = np.digitize(actual_scores, bins)

    bin_stats = []
    for bin_idx in range(1, len(bins)):
        in_bin = (indices == bin_idx)
        count_in_bin = in_bin.sum()
        if count_in_bin == 0:
            bin_stats.append({
                "fidelity_bin_start": bins[bin_idx - 1],
                "fidelity_bin_end": bins[bin_idx],
                "mean_abs_error": np.nan,
                "count": 0
            })
        else:
            mean_err = errors[in_bin].mean()
            bin_stats.append({
                "fidelity_bin_start": bins[bin_idx - 1],
                "fidelity_bin_end": bins[bin_idx],
                "mean_abs_error": mean_err,
                "count": int(count_in_bin)
            })

    df_fidelity_error = pd.DataFrame(bin_stats)
    df_fidelity_error["bin_mid"] = (df_fidelity_error["fidelity_bin_start"] + df_fidelity_error["fidelity_bin_end"]) / 2

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=df_fidelity_error,
        x="bin_mid",
        y="mean_abs_error",
        marker="o",
        color="salmon",
        label="Mean Abs Error"
    )
    ax.set_xlabel("Actual Fidelity (Binned)", fontsize=14)
    ax.set_ylabel("Mean Absolute Error", fontsize=14)
    ax.set_title(f"Mean Error vs. Actual Fidelity (in {num_bins} bins)", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if output_dir:
        plot_path = os.path.join(output_dir, "error_by_fidelity.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved error-by-fidelity plot to: {plot_path}")

        csv_path = os.path.join(output_dir, "error_by_fidelity.csv")
        df_fidelity_error.to_csv(csv_path, index=False)
        print(f"[INFO] Saved error-by-fidelity stats to: {csv_path}")

    return df_fidelity_error