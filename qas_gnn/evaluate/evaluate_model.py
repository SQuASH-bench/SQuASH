import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

from config import DeviceConfig, get_model_config, PathConfig, QCConfig
from evaluate.evaluate_utils import load_rf_model, evaluate_rf, load_gnn_model, evaluate_gnn, analyze_circuits, \
    analyze_error_by_fidelity, analyze_gate_features, compute_metrics
from util.data_loader import load_data



if __name__ == "__main__":
    # --- Common Setup ---
    gate_set = 'gs2'
    data_set_name = 'test_dataset_without_aug_gs2' # f'{gate_set}'
    model_name = 'train_dataset_augmented_gs2' # f'{gate_set}'

    color = 'skyblue'

    gnn = False
    rf = True

    print(f"=== Evaluation on {data_set_name} ===")

    device_config = DeviceConfig()
    device = device_config.device
    path_config = PathConfig()
    if gate_set == 'gs1':
        gate_mapping = QCConfig().gate_set_1
    elif gate_set == 'gs2':
        gate_mapping = QCConfig().gate_set_2
    elif gate_set == 'ml':
        gate_mapping = QCConfig().gate_set_ml


    if rf:
        # === Random Forest Evaluation ===
        print("=== Random Forest Evaluation ===")
        model_name_rf = f"random_forest_{gate_set}"
        model_config_rf = get_model_config(model_name_rf, device)

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
            color="orange",
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


    if gnn:
        # === GNN Evaluation ===
        print("=== GNN Evaluation ===")
        print(f'gcn_{model_name}')
        model_config_gnn = get_model_config(f'gcn_{gate_set}', device)

        # Load dataset
        circuits = load_data(os.path.join(path_config.paths['test_data'], f'{data_set_name}.pt'))
        print(f"Loaded {len(circuits)} circuits for GNN evaluation.")
        if len(circuits) == 0:
            raise ValueError("No data available for GNN evaluation.")

        # # Update config with data dimensions
        # model_config_gnn.edge_attr_dim = circuits[0].edge_attr.size(1)
        # model_config_gnn.num_node_features = circuits[0].x.size(1)

        model_path_gnn = os.path.join(path_config.paths['trained_models'], f"gcn_{model_name}.pth")
        gnn_model = load_gnn_model(model_path_gnn, model_config_gnn)
        predicted_gnn, actual_gnn = evaluate_gnn(circuits, gnn_model, model_config_gnn)
        metrics_gnn = compute_metrics(predicted_gnn, actual_gnn, label="GNN")
        errors_gnn = predicted_gnn - actual_gnn

        # # --- Additional GNN Analysis ---
        analyze_circuits(
            circuits=circuits,
            predicted=predicted_gnn,
            actual=actual_gnn,
            gate_mapping=gate_mapping,
            top_percent=5.0,
            min_fidelity=None,
            output_dir=path_config.paths['benchmark'],
            file_prefix="problem_circuits_all",
            color=color,
            model_name=model_name,
        )
        analyze_circuits(
            circuits=circuits,
            predicted=predicted_gnn,
            actual=actual_gnn,
            gate_mapping=gate_mapping,
            top_percent=5.0,
            min_fidelity=0.9,
            output_dir=path_config.paths['benchmark'],
            file_prefix="problem_circuits_hifid"
        )
        gate_feature_stats = analyze_gate_features(
            circuits=circuits,
            predicted_scores=predicted_gnn,
            actual_scores=actual_gnn,
            gate_mapping=gate_mapping,
            output_dir=path_config.paths['benchmark']
        )

        print("\nGate Feature Stats (presence vs. error):")
        print(gate_feature_stats.head(10))

        df_fidelity_error = analyze_error_by_fidelity(
            actual_scores=actual_gnn,
            predicted_scores=predicted_gnn,
            num_bins=10,
            output_dir=path_config.paths['benchmark']
        )
        print("\nAverage error by fidelity bin:")
        print(df_fidelity_error)

        df_gnn = pd.DataFrame({
            "Actual Fidelity": actual_gnn,
            "Predicted Fidelity": predicted_gnn,
            "Error": errors_gnn
        })


        # --- GNN Plots ---

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(1, 1, figsize=(18, 5))

        # Error Distribution Histogram
        sns.histplot(
            data=df_gnn,
            x="Error",
            bins=30,
            color=color,
            ax=ax
        )
        ax.set_title("Error Distribution (Predicted - Actual)", fontsize=16)
        ax.set_xlabel("Error", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)

        formatter = FuncFormatter(lambda x, pos: f"{int(x / 1000)}T")
        ax.yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.savefig(os.path.join(path_config.paths['plots'], f"gcn_{model_name}_error.png"))
        plt.show()


        # ---------

        sns.set_theme(style="darkgrid")

        g = sns.JointGrid(
            data=df_gnn,
            x="Actual Fidelity",
            y="Predicted Fidelity",
            marginal_ticks=True
        )

        g.ax_joint.set(yscale="log")

        cax = g.figure.add_axes([.15, .55, .02, .2])

        g.plot_joint(
            sns.histplot,
            discrete=(True, False),
            cmap="Blues",
            pmax=.8,
            cbar=True,
            cbar_ax=cax
        )

        if g.ax_joint.collections:
            g.ax_joint.collections[0].set_norm(mcolors.LogNorm())

        g.plot_marginals(sns.histplot, element="step", color="blue")
        plt.show()


        # ---------

        sns.set_theme(style="darkgrid")

        hex_plot = sns.jointplot(
            data=df_gnn,
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

        hex_plot.ax_joint.set_ylim(-0.01, 1.01)
        hex_plot.ax_joint.set_xlim(-0.01, 1.01)

        hex_plot.set_axis_labels("Actual Fidelity", "Predicted Fidelity", fontsize=14)
        hex_plot.fig.suptitle("Fidelity Density (Log Scale)", fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(path_config.paths['plots'], f"gcn_{model_name}_distr.png"))
        plt.show()




